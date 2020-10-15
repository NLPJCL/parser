# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
#from utils.fn import stripe


class MatrixTree(nn.Module):
    r"""
    MatrixTree for calculating partition functions and marginals in :math:`O(n^3)` for directed spanning trees
    (a.k.a. non-projective trees) by an adaptation of Kirchhoff's MatrixTree Theorem.
    It differs from the original paper in that marginals are computed via back-propagation
    rather than matrix inversion.
    References:
        - Terry Koo, Amir Globerson, Xavier Carreras and Michael Collins. 2007.
          `Structured Prediction Models via the Matrix-Tree Theorem`_.
    .. _Structured Prediction Models via the Matrix-Tree Theorem:
        https://www.aclweb.org/anthology/D07-1015/
    """

    @torch.enable_grad()
    def forward(self, scores, mask,i,target=None,partial=False,mbr=False):
        r"""
        Args:
            scores (~torch.Tensor): ``[batch_size, seq_len, seq_len]``.
                Scores of all possible dependent-head pairs.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask to avoid aggregation on padding tokens.
                The first column serving as pseudo words for roots should be ``False``.
            target (~torch.LongTensor): ``[batch_size, seq_len]``.
                The tensor of gold-standard dependent-head pairs. Default: ``None``.
            mbr (bool):
                If ``True``, marginals will be returned to perform minimum Bayes-risk (MBR) decoding. Default: ``False``.
        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first is the training loss averaged by the number of tokens, which won't be returned if ``target=None``.
                The second is a tensor of shape ``[batch_size, seq_len, seq_len]``, in which are marginals if ``mbr=True``,
                or original scores otherwise.
        """
        self.i=i
        training = scores.requires_grad
        # double precision to prevent overflows
        scores = scores.double()
        #logZ = self.matrix_tree(scores.requires_grad_(), mask)
        Z_L,Z_m,Z_lens = self.matrix_tree(scores.requires_grad_(), mask)
        probs = scores
        # calculate the marginals
        if mbr:
            probs, = autograd.grad(logZ, probs, retain_graph=training)
        probs = probs.float()
        if target is None:
            return probs
        target[:,0]=0#把第一个标签去掉。(因为第一个标签是随便填的)这里是因为加载的时候不对，所以导致了这样。

        if partial:
            z_L,z_m,z_lens=self.matrix_tree(scores, mask,target)
        else:
            score = scores.gather(-1, target.unsqueeze(-1)).squeeze(-1)[mask].sum()
        #score = scores.gather(-1, target.unsqueeze(-1)).#得到真实的分数。
        #.squeeze(-1)[mask].sum()    去掉最后一维没用的东西，mask掉第一个词（每一个词补充的到root的分数），然后再把真实的句子拿出来求和。
        with torch.no_grad():
            Z_mask=Z_L[:, 1:, 1:].det().ne(0)
            z_mask=z_L[:, 1:, 1:].det().ne(0)
            u_mask=Z_mask&z_mask
        logZ=(Z_L[:,1:,1:].slogdet()[1][u_mask]+Z_m[u_mask]*Z_lens[u_mask]).sum()

        score=(z_L[:,1:,1:].slogdet()[1][u_mask]+z_m[u_mask]*z_lens[u_mask]).sum()

        loss = (logZ - score).float() / mask.sum()
        return loss, probs

    def matrix_tree(self, scores, mask,cands=None):

        lens = mask.sum(-1)
        batch_size, seq_len, _ = scores.shape
        mask = mask.index_fill(1, lens.new_tensor(0).long(), 1)#给第一维添加1

        if cands is not None:
            mask_ = (mask.unsqueeze(1) & mask.unsqueeze(-1))#生成一个批次的句子。
            
            cands = cands.unsqueeze(-1).index_fill(1, lens.new_tensor(0), -1)#给第一维添加-1,所以第一个都是-1.
            #[batch_size, seq_len,1] 
            cands = cands.eq(lens.new_tensor(range(seq_len))) | cands.lt(0)
            #cands.lt(0) 没有弧的是True，并且第一个也是-1.的为真。
            #cands.eq(lens.new_tensor(range(seq_len))) #batch x sen_len xsen_len     真的有那条弧的为真
            cands = cands& mask_
            #mask
            #scores = scores.masked_fill(~cands, torch.finfo().min) 
            scores = scores.masked_fill(~cands, float('-inf'))

        #w=mask.unsqueeze(-1) & mask.unsqueeze(-2)
        # w.index_fill_(2,scores.new_tensor(0).long(),1)
        #scores = scores.masked_fill(~w, float('-inf'))
        scores = scores.masked_fill(~(mask.unsqueeze(-1) & mask.unsqueeze(-2)), float('-inf'))#(为了把填充的部分的分数抹掉)
        # the numerical stability trick is borrowed from timvieira (https://github.com/timvieira/spanning_tree)
        # log(det(exp(M))) = log(det(exp(M - m) * exp(m)))
        #                  = log(det(exp(M - m)) * exp(m)^n)
        #                  = log(det(exp(M - m))) + m*n
        m = scores.view(batch_size, -1).max(-1)[0]#找每个batch中分数最大的那个。
        # clamp the lower bound to `torch.finfo().tiny` to prevent underflows
        #A = torch.exp(scores - m.view(-1, 1, 1)).clamp(torch.finfo().tiny)#[batch_size, seq_len, seq_len]
        A = torch.exp(scores - m.view(-1, 1, 1)).clamp(torch.finfo().tiny)
        # D is the weighted degree matrix
        # D(i, j) = sum_j(A(i, j)), if h == m
        #           0,              otherwise
        A.masked_fill_(~mask.unsqueeze(-1), 0)
        
        D = torch.zeros_like(A)
        D.diagonal(0, 1, 2).copy_(A.sum(-1))
        # Laplacian matrix
        
        L = nn.init.eye_(torch.empty_like(A[0])).repeat(batch_size, 1, 1)#构造每一个句子的对角线都为1的一个batch。

        L = L.masked_scatter_(mask.unsqueeze(-1), (D - A)[mask])#在mask为真的地方，赋值D-A的值。
        # calculate the partition (a.k.a normalization) term
        # Z = L^(0, 0), which is the minor of L w.r.t row 0 and column 0
        #L.masked_fill_(~mask.unsqueeze(1), 0)#把列pad的地方补为-inf，为了不让计算loss的时候带入，而行，因为mask的出现并不会出现。
        #logZ=m*lens.sum()

        #with torch.no_grad():
        #z=L[:,1:,1:].slogdet()[1]

        '''
        z=L.detach()
        z=z[:, 1:, 1:].det()
        z_mask=z.ne(0)
        
        logZ=(L[:,1:,1:].slogdet()[1][z_mask]+m[z_mask]*lens[z_mask]).sum()
        '''
        #random_mask=L.new_ones(L.size(0)).bernoulli_(0.8).ne(0)
        #random_mask
        return L,m,lens
        #logZ=(L[:,1:,1:].slogdet()[1][random_mask]+m[random_mask]*lens[random_mask]).sum()

        #logZ = (L[:, 1:, 1:].slogdet()[1] + m*lens).sum()
        #return logZ,random_mask

if __name__=='__main__':
    s=MatrixTree()
    '''
    scores = torch.tensor([[[-11.9436, -13.1464, -6.4789, -13.8917],
                            [-60.6957, -60.2866, -48.6457, -63.8125],
                            [-38.1747, -49.9296, -45.2733, -49.5571],
                            [-19.7504, -23.9066, -9.9139, -16.2088]]]).long()
    '''
    scores=torch.ones(4,4,4).float()
    scores.requires_grad_()
    mask = torch.tensor([[True, True, True,False],
                         [True, True, False,False],
                         [True, True, True,False],
                         [True,True,False,False]])
    mask[:,0]=False
    target=torch.Tensor([[-1,2,-1,0,],
                        [-1,2,0,0],
                        [-1,2,-1,0],
                        [-1,0,0,0]])

    #print(scores&mask)

    loss,_=s(scores,mask,target,True)
