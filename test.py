import torch.nn as nn
import torch
from datetime import datetime, timedelta

# an Embedding module containing 10 tensors of size 3
'''
x=torch.ones(2,2,2)

mask=torch.tensor([-1,0,1,2])
mask=mask.view((2,1,2))

mask=mask.ge(0)
print(mask)
s=x.masked_fill_(mask,5)

print(s)


'''
'''
y=torch.ones(50,26,26)

print(y[0][:5,:5].size())
'''
'''
a=torch.randn(2,3,3)
print(a)
print(a.diagonal(0,-2,-1))
print(a.diagonal(0,1,2))
'''

'''
s=torch.tensor([[True,False],
                [False,True]])
print(s)
print(~s)
'''

'''
s=torch.arange(10).view(2,5)

print(s[torch.tensor([0,1]),torch.tensor([0,2])])
print(s)

'''

'''
t = torch.tensor([[1,2],[3,4]])
print(torch.gather(t, 1, torch.tensor([[0,0],[1,0]])))
'''
#(0,0)(0,0)(1,1)(1,0)
'''
print(torch.gather(t, 0, torch.tensor([[0,0],[1,0]])))
#(0,0)(0,1)(1,0)(0,1)

'''

'''
s=torch.Tensor([[1,2],[3,4]])

print(s.new_tensor(0))
print(s.new_tensor(0).long())


mask=s.index_fill(1,s.new_tensor(1).long(),1)
print(mask)

'''

##print(torch.finfo().tiny)
score=torch.ones(4,4,4)

mask_=torch.Tensor([ [True,True,True,False],
                    [True,True,True,False],
                    [True,True,True,False],
                    [True,True,True,False]]).bool()

mask_[:,0]=0
w=mask_.unsqueeze(-1) & mask_.unsqueeze(-2)
w.index_fill_(2,score.new_tensor(0).long(),1)
print(w)

score = score.masked_fill(~w,float('-inf'))

#score = score.masked_fill(~(mask_.unsqueeze(-1) & mask_.unsqueeze(-2)), float('-inf'))#(为了把填充的部分的分数抹掉)

print(score)


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





