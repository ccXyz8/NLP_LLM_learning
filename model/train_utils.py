import torch
import torch.nn as nn
from torch.autograd import Variable

class MultiGPULossCompute:
    def __init__(self, generator, criterion, devices,opt=None,chunk_size=5):
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out,target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets,target_gpus=self.devices)

        chunk_size = self.chunk_size
        for i in range(0,out_scatter[0].size(1),chunk_size):
            out_column = [[Variable(o[: i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]

            gen = nn.parallel.parallel_apply(generator,out_column)

            y = [(g.contiguous().view(-1,g.size(-1)),
                  t[:,i:i+chunk_size].contiguous().view(-1))
                  for g,t in zip(gen,targets)]
            loss = nn.parallel.parallel_apply(self.criterion,y)

            l_ = nn.parallel.gather(loss,target_device=self.devices[0])
            l_ = l_.sum() / normalize
            total+=l_

            if self.opt is not None:
                l_.backward()
                for j,l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        if self.opt is not None:
            out_grad = [Variable(torch.cat(og,dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1 .backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize

class NoamOpt:
    def __init__(self,model_size,facter,warmup,optimizer):
        self.model_size = model_size
        self._step=0
        self.facter = facter
        self.warmup = warmup
        self.optimizer = optimizer
        self._rate=0

    def step(self):
        self._step+=1
        rate = self.rate()

        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self,step=None):
        if step is None:
            step+=self._step
        return self.facter *(self.model_size ** (-0.5) *min(step ** (-0.5),step*self.warmup ** (-1.5)))

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model,1,10000,
                   torch.optim.Adam(model.parameters(),lr=0,betas=(0.9,0.98),eps=1e-9))
