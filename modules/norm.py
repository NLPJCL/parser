import  torch.nn as nn
import torch

class Norm(nn.Module):
    def forward(self,input,mask):
            lens=mask.sum(-1).unsqueeze(-1)#(3,2) 2    #lens=mask.sum(-1).unsqueeze(-1).unsqueeze(-1)
            
            t=mask.unsqueeze(-1)&mask.unsqueeze(-2)
            
            input.masked_fill_(~t,0)#2x3x3
            
            mean=(input.sum(-1)/lens).unsqueeze(-1)#2x3矩阵。->2x3x1
            
            std=torch.sqrt(torch.pow(input-mean,2).masked_fill_(~t,0).sum(-1)/lens).unsqueeze(-1)#2x3 ->2x3x1
            
            return ((input-mean)/(std+1e-6)).masked_fill_(~mask.unsqueeze(1),float('-inf')) 
