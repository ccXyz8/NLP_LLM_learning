import torch
import config
import logging
import numpy as np
from tools.tokenizer_utils import english_tokenizer_load
from model.tf_model import make_model
from tools.tokenizer_utils import chinese_tokenizer_load
from beam_decoder import beam_search

logging.basicConfig(format='%(asctime)s-%(name)s-%(levelname)s-%(message)s-%(funcName)s:%(lineno)d', level=logging.INFO)

def translate(src,model):
    sp_chn = chinese_tokenizer_load()

    with torch.no_grad():
        model.load_state_dict(torch.load(config.test_model_path))
        model.eval()

        src_mask = (src!=0).unsqueeze(-2)

        decode_result ,_ = beam_search(
            model,
            src,
            src_mask,
            config.max_len,
            config.padding_idx,
            config.bos_idx,
            config.eos_idx,
            config.beam_size,
            config.device
        )

        decode_result = [h[0] for h in decode_result]

        translation = [sp_chn.decode_ids(_s) for _s in decode_result]

        return translation[0]

def one_sentence_translate(sent):
    model = make_model(
        config.src_vocab_size,
        config.tgt_vocab_size,
        config.n_layers,
        config.d_model,
        config.d_ff,
        config.n_heads,
        config.dropout
    )

    BOS = english_tokenizer_load().bos_id()
    EOS = english_tokenizer_load().eos_id()

    src_tokens =[[BOS]+english_tokenizer_load().EncodeAsIds(sent)+[EOS]]

    batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)

    return translate(batch_input,model)

def translate_example():
    while True:
        sent = input("请输入英文句子进行翻译: ")
        translation = one_sentence_translate(sent)
        print("翻译结果：", translation)

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import warnings
    warnings.filterwarnings("ignore")
    translate_example()