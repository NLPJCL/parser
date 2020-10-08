import torch.nn as nn
import torch
from datetime import datetime, timedelta

# an Embedding module containing 10 tensors of size 3

def norm(input,mask):
        lens=mask.sum(-1)
        for i in range(input.size(0)):
            w=input[i][:lens[i],:lens[i]].view(lens[i], lens[i])
            #w=input[i][mazzsk[i].unsqueeze(-1) & mask[i].unsqueeze(-2)].view(lens[i], -1)
            #gamma = torch.ones(w.size(-1))
            #beta = torch.zeros(w.size(-1))
            eps = 1e-6
            mean = w.mean(-1, keepdim=True)
            std = w.std(-1, unbiased=True, keepdim=True)
            w.sub_(mean)
            w.div_(std+eps)
            #w=(w - mean) / (std + eps)
            #w = nn.Softmax(-1)(w)
            #input[i, 0:w.size(0), 0:w.size(1)].copy_(w)
##print(torch.finfo().tiny)



'''
input=torch.arange(2*3*3).view(2,3,3).float()

mask=torch.Tensor([[True,True,True],
                  [True,True,False]]).bool()
norm(input,mask)

print(input)
'''


x = torch.tensor([[1., -1.], [1., 1.]], requires_grad=True)

w=x[0,:]

print(w._version)
z=w.std(-1, unbiased=True, keepdim=True)
print(w._version)

'''
out = x.pow(2)

w=out[0,:]
w.add_(1)

print(out)

out =out.pow(2)

print(out)
print(out.is_leaf)
out=out.sum()
print(out)
out.backward()	
print(x.grad)
'''

'''
s_arc=torch.ones(4,4,4)

mask=torch.Tensor([ [True,True,True,False],
                    [True,True,False,False],
                    [True,True,False,False],
                    [True,True,True,False]]).bool()

s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))#把列pad的地方补为-inf，为了不让计算loss的时候带入，而行，因为mask的出现并不会出现。


print(s_arc)
'''

'''
mask_[:,0]=0
w=mask_.unsqueeze(-1) & mask_.unsqueeze(-2)
w.index_fill_(2,score.new_tensor(0).long(),1)
print(w)

score = score.masked_fill(~w,float('-inf'))

#score = score.masked_fill(~(mask_.unsqueeze(-1) & mask_.unsqueeze(-2)), float('-inf'))#(为了把填充的部分的分数抹掉)

print(score)
'''

'''
t=mask_.unsqueeze(1)
u=mask_.unsqueeze(-1)
(mask_.unsqueeze(1) & mask_.unsqueeze(-1))#生成一个批次的句子。
print((mask_.unsqueeze(1) & mask_.unsqueeze(-1)))
'''
#print((mask_.unsqueeze(-1) & mask_.unsqueeze(-2)))


#print(torch.masked_fill(score,~(mask_.unsqueeze(1) & mask_.unsqueeze(-1)),-float('-inf')))

#print(torch.masked_fill(score,~mask_.unsqueeze(1),-float('-inf')))

'''
b = torch.Tensor([[[1,1],
                    [2,2]],

                    [[3,3],
                    [4,4]]])
mask=torch.Tensor([ [False,True],
                    [False,True]]).bool()

#mask[:,0]=False
print(b[mask])



w=torch.Tensor([ [2,2],
                    [1,1]])

print(w+b)

c=torch.Tensor([True,False]).bool()

print(c.size())
print(w[c])

q=torch.Tensor([[True,False],
                [True,False]]).bool()

print(w[q])

print(w+torch.Tensor([(1,2)]))
'''

'''
c=torch.tensor([1,2,3])
#print(b-c)

print(b)

print(b-c.view(-1,1))
'''


'''
mask.ge_(0)

print(mask.unsqueeze(-1))
print(mask.unsqueeze(-2))
print(mask.unsqueeze(-1)&mask.unsqueeze(-2))

print(~(mask.unsqueeze(-1)&mask.unsqueeze(-2)))

#s=s.masked_fill(~(mask.unsqueeze(-1)&mask.unsqueeze(-2)),8)

print(s)

'''





