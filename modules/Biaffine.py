import  torch.nn as nn
import torch
class Biaffine(nn.Module):

    def __init__(self,n_input,n_output=1):
        super().__init__()

        self.n_in=n_input
        self.n_out=n_output
        self.weight=nn.Parameter(torch.Tensor(self.n_out,self.n_in+1,self.n_in))#1x501x500
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.weight)
    def forward(self,x,y):
        '''
        :param  x:mlp_d [batch_size, seq_len, n_in]
        :param  y:mlp_h [batch_size, seq_len, n_in]
        :return:s:[batch_size, n_out, seq_len, seq_len]
                If n_out is 1, size of n_out will be squeezed automatically.
        '''
        x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        #x*w*Y^T 求的是s[i][j]j->i的弧分数
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        s = s.squeeze(1)
        return s








