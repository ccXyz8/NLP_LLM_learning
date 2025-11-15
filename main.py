import os

import config
import torch
from torch.utils.data import DataLoader

from config import padding_idx
from tools.data_loader import MTDataset
from model.tf_model import make_model
import logging
import sacrebleu
from tqdm import tqdm

from beam_decoder import beam_search
from model.train_utils import MultiGPULossCompute,get_std_opt
from tools.tokenizer_utils import chinese_tokenizer_load
from tools.create_exp_folder import create_exp_folder

logging.basicConfig(format='%(asctime)s-%(name)s-%(levelname)s-%(message)s', level=logging.INFO)

def run_epoch(data,model,loss_compute):
    total_tokens =0.
    total_loss = 0.

    for batch in tqdm(data):
        out =model(batch.src,batch.trg,batch.src_mask,batch.trg_mask)
        loss = loss_compute(out,batch.trg_y,batch.ntokens)

        total_loss += loss
        total_tokens += batch.ntokens

    return total_loss/total_tokens

def train(train_data,dev_data,model,model_par,criterion,optimizer):
    best_bleu_score = -float("inf")
    exp_folder,weights_folder = create_exp_folder()

    for epoch in range(1,config.epoch_num+1):
        logging.info(f"第{epoch}轮模型训练与验证")
        model.train()

        train_loss = run_epoch(train_data,model_par,
                               MultiGPULossCompute(model.generator,criterion,config.device_id,optimizer))

        model.eval()

        dev_loss = run_epoch(dev_data,model_par,
                             MultiGPULossCompute(model.generator,criterion,config.device_id,optimizer))

        bleu_score = evaluate(dev_data,model)
        logging.info(f"Epoch: {epoch}, train_loss: {train_loss:.3f}, val_loss: {dev_loss:.3f}, Bleu Score: {bleu_score:.2f}\n")

        if bleu_score > best_bleu_score:
            if best_bleu_score != -float("inf"):
                old_model_path = f"{weights_folder}/best_bleu_{best_bleu_score:.2f}.pth"
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)

            model_path_best = f"{weights_folder}/best_bleu_{bleu_score:.2f}.pth"
            torch.save(model.state_dict(),model_path_best)
            best_bleu_score = bleu_score

        if epoch == config.epoch_num:
            model_path_last = f"{weights_folder}/last_bleu_{bleu_score:.2f}.pth"  # 构建模型保存路径，包含BLEU分数
            torch.save(model.state_dict(),model_path_last)

def evaluate(data,model):
    sp_chn = chinese_tokenizer_load()
    trg=[]
    res=[]

    with torch.no_grad():
        for batch in tqdm(data):
            cn_sent = batch.trg_text
            src = batch.src
            src_mask = (src!=0).unsqueeze(-2)

            decode_result,_=beam_search(model,src,src_mask,config.max_len,
                                        config.padding_idx,config.bos_idx,config.eos_idx,
                                        config.beam_size,config.device)

            decode_result = [h[0] for h in decode_result]

            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            trg.append(cn_sent)
            res.append(translation)

    trg=[trg]
    bleu = sacrebleu.corpus_bleu(trg,res,tokenize='zh')
    return float(bleu.score)

def test(data,model,criterion):
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model_par = torch.nn.DataParallel(model)
        model.eval()

        test_loss = run_epoch(data, model_par,
                              MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        bleu_score = evaluate(data, model,'test')
        logging.info('Test loss: {},  Bleu Score: {}'.format(test_loss, bleu_score))

def run():
    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset,shuffle=True,batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset,shuffle=True,batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    model = make_model(config.src_vocab_size,config.tgt_vocab_size,config.n_layers,
                       config.d_model,config.d_ff,config.n_heads,config.dropout)

    model_par = torch.nn.DataParallel(model)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0,reduction='sum')

    optimizer = get_std_opt(model)
    train(train_dataloader,dev_dataloader,model,model_par,criterion,optimizer)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import warnings
    warnings.filterwarnings('ignore')
    run()