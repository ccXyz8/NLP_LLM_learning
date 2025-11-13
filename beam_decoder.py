import torch
from tools .data_loader import subsequent_mask

class Beam:
    def __init__(self, size,pad,bos,eox,device=False):
        self.size = size
        self._done=False
        self.PAD = pad
        self.BOS = bos
        self.EOX = eox

        self.scores = torch.zeros((size,),dtype=torch.float,device=device)
        self.all_scores =[]

        self.prev_ks=[]
        self.next_ys = [torch.full((size,),self.PAD,dtype=torch.long,device=device)]
        self.next_ys[0][0] = self.BOS

    def get_current_states(self):
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def advance(self,word_logprob):
        num_words = word_logprob.size(1)

        if len(self.prev_ks)>0:
            beam_lk = word_logprob + self.scores.unsqueeze(1).expand_as(word_logprob)
        else:
            beam_lk = word_logprob[0]

        fat_beam_lk = beam_lk.view(-1)
        best_scores,best_scores_id = fat_beam_lk.topk(self.size,0,True,True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        if self.next_ys[-1][0].item() == self.EOX:
            self.done = True
            self.all_scores.append(self.scores)

        return self._done

    def sort_scores(self):
        return torch.sort(self.scores,0,True)

    def get_the_best_score_and_idx(self):
        scores,ids = self.sort_scores()
        return scores[0],ids[0]

    def get_tentative_hypothesis(self):
        if len(self.next_ys) == 1:
            dex_seq = self.next_ys[0].unsqueeze(1)
        else:
            _,keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)

        return dec_seq

    def get_hypothesis(self, k):
        hyp = []
        for j in range(len(self.prev_ks)-1,-1,-1):
            hyp.append(self.next_ys[j+1][k])
            k = self.prev_ks[j][k]

        return list(map(lambda x:x.item(),hyp[::-1]))

def beam_search( model ,src,src_mask,max_len,pad,bos,eox,beam_size,device):

    def get_inst_idx_to_tensor_position_map(inst_idx_list):
        return {inst_idx :tensor_position for tensor_position,inst_idx in enumerate(inst_idx_list)}

    def collect_active_part(beamed_tensor ,curr_active_inst_idx ,n_prev_active_inst,n_bm):
        _,*d_hs = beamed_tensor.size()
        n_curr_active_inst = len(curr_active_inst_idx)
        new_shape = (n_curr_active_inst,*d_hs)

        beamed_tensor = beamed_tensor.view(n_prev_active_inst,-1)
        beamed_tensor = beamed_tensor.index_select(0,curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(*new_shape)

        return beamed_tensor



