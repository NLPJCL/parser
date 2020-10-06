from utils import mst,chuliu_edmonds
import  torch
if __name__=='__main__':
    t='-inf'
    print(t)
    s=torch.tensor([[float('-inf'),9,10,9],
                    [float('-inf'),float('-inf'),20,3],
                    [float('-inf'),30,float('-inf'),30],
                    [float('-inf'),11,float('-inf'),float('-inf')]])
    #print(s)
    s.transpose_(0,1)
    print(s)
    t=chuliu_edmonds(s)