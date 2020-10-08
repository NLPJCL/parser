from datetime import datetime, timedelta
import torch
import torch.nn as nn
from modules import Biaffine,MatrixTree
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from utils import mst
from torch.optim.lr_scheduler import ExponentialLR


#CRF_Biaffine

def norm(input,mask):
        lens=mask.sum(-1)
        for i in range(input.size(0)):
            w=input[i][:lens[i],:lens[i]].view(lens[i], lens[i])
            #w=input[i][mazzsk[i].unsqueeze(-1) & mask[i].unsqueeze(-2)].view(lens[i], -1)
            #gamma = torch.ones(w.size(-1))
            #beta = torch.zeros(w.size(-1))
            eps = 1e-6
            with torch.no_grad():
                mean = w.mean(-1, keepdim=True)
                std = w.std(-1, unbiased=True, keepdim=True)
            w.sub_(mean)
            w.div_(std+eps)

class CRF_Biaffine(nn.Module):
    def __init__(self,word_embed,tag_embed,args):
        super(CRF_Biaffine, self).__init__()
        self.word_embed=nn.Embedding.from_pretrained(word_embed,False)
        self.tag_embed=nn.Embedding.from_pretrained(tag_embed,False)
        #lstm层
        self.lstm=nn.LSTM(input_size=args.n_embed+args.n_tag_embed,
                          hidden_size=args.n_lstm_hidden,
                          num_layers=args.n_lstm_layers,
                          bidirectional=True)
        #两个mlp层
        self.mlp_arc_d = nn.Linear(args.n_lstm_hidden * 2,
                             args.n_mlp_arc)

        self.mlp_arc_h = nn.Linear(args.n_lstm_hidden * 2,
                             args.n_mlp_arc)
        #数据归一化层
        #self.layer_norm=torch.nn.LayerNorm(args.n_embed+args.n_tag_embed)#lstm层的dropout
        #self.layer_norm_d=torch.nn.LayerNorm(args.n_mlp_arc)
        #self.layer_norm_h=torch.nn.LayerNorm(args.n_mlp_arc)

        #biaffine层
        self.arc_attn=Biaffine(n_input=args.n_mlp_arc,n_output=1)
        #drop层
        self.dropout_input= nn.Dropout(0.1)
        self.dropout_lstm=nn.Dropout(0.4)
        #normalization层
       # self.normalization=nn.BatchNorm1d()


    def forward(self,words,feats):
        '''

        :param words:[batch_size,seq_len]
        :param feats: [batch_size,seq_len]
        :return:
        '''

        mask=words.ne(0)
        lens=mask.sum(1)
        #[batch_size,seq_len,100]
        word_embed=self.word_embed(words)
        #[batch_size,seq_len,50]
        feat_embed=self.tag_embed(feats)
        #对输入层dropout
        word_embed=self.dropout_input(word_embed)
        feat_embed=self.dropout_input(feat_embed)

        #[batch_size,seq_len,150]
        embed = torch.cat((word_embed, feat_embed), -1)

        #x=self.layer_norm(embed)
        
        x = pack_padded_sequence(embed, lens, True, False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.dropout_lstm(x)

        arc_d=self.mlp_arc_d(x)
        arc_h=self.mlp_arc_h(x)


        #arc_d=self.layer_norm_d(arc_d)
        #arc_h=self.layer_norm_h(arc_h)

        s_arc=self.arc_attn(arc_d,arc_h)
        s_arc.masked_fill_(~mask.unsqueeze(1), float('-inf'))#把列pad的地方补为-inf，为了不让计算loss的时候带入，而行，因为mask的出现并不会出现。
        return s_arc

    with torch.autograd.set_detect_anomaly(True):
        def update(self,train_loader):
            # 设置为训练模式
            self.train()
            for words, feats, arcs, rels,lens in train_loader:
                # 清楚梯度
                self.optimizer.zero_grad()

                # 获取掩码
                mask = words.ne(0)
                # ignore the first token of each sentence
                # 前向计算。
                s_arc = self.forward(words, feats)

                norm(s_arc,mask) #加归一化。

                mask[:,0] = 0#ljc(计算分数时，用target取score中的分数，本来就不会有第一行的分数，也就是其他词到root的分数)

                loss,_=self.loss_function(s_arc, mask,arcs,self.args.partial)
                # 计算梯度
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                # 更新参数
                self.optimizer.step()
                self.scheduler.step()

    def decode(self,s_arc, mask):

        lens = mask.sum(1)
        # prevent self-loops
        s_arc.diagonal(0, 1, 2).fill_(float('-inf'))
        arc_preds = mst(s_arc, mask)
        return arc_preds

    def evaluate(self,loader):
        self.eval()
        total_loss= 0
        total_corr,total_all=0,0,
        for words, feats, arcs, rels,lens in loader:
            mask = words.ne(0)
            # ignore the first token of each sentence
            s_arc= self.forward(words, feats)
            norm(s_arc,mask)#评测也加归一化
           # loss = self.loss(s_arc, arcs, mask)
            mask[:, 0] = 0
            arc_preds = self.decode(s_arc, mask)
           # total_loss += loss.item()
            #mask &= words.unsqueeze(-1).ne(puncts).all(-1)
            if self.args.partial:
                mask&=arcs.ge(0)
            targets = list(torch.split(arc_preds[mask], mask.sum(1).tolist()))
            arcs = list(torch.split(arcs[mask], mask.sum(1).tolist()))

            for i in range(len(targets)):
                total_corr += torch.sum(arcs[i] == targets[i]).item()
                total_all+=len(targets[i])
        print("correc/total={}/{} ".format(total_corr,total_all))
        total_loss /= len(loader)

        return total_loss,total_corr/total_all

    def fit(self,args,train_loader,dev_loader,test_loader,f,device):
        total_time = timedelta()
        # 记录最大迭代次数和准确率
        max_test_epoch, max_test = 0, 0.0
        max_dev_epoch, max_dev = 0, 0.0

        self.device=device
        self.args=args
        self.loss_function = MatrixTree()

        self.optimizer = optim.Adam(self.parameters(), lr=2e-3, betas=(0.9, 0.9), eps=1e-12)
        self.scheduler = ExponentialLR(self.optimizer, .75**(1/5000))

        for epoch in range(1, args.epochs + 1):
            start = datetime.now()
            self.update(train_loader)
            print("Epoch:{}/{}".format(epoch, args.epochs))

            f.write("Epoch:{}/{}\n".format(epoch, args.epochs))
            # 评估部分。
            '''
            print("Train:")
            f.write("Train:")
            train_loss, train_precision, train_recall, train_F = self.evaluate(train_loader)

            print("Loss:{}  Pricision:{}  Recall:{}  F:{}".format(train_loss, train_precision, train_recall, train_F))

            f.write(
                "Loss:{}  Pricision:{}  Recall:{}  F:{}\n".format(train_loss, train_precision, train_recall, train_F))
            '''
            print("Dev:")
            f.write("Dev:")

            dev_loss, dev_uas = self.evaluate(dev_loader)
            print("Loss:{}  UAS:{} ".format(dev_loss, dev_uas))
            f.write("Loss:{}  UAS:{}\n".format(dev_loss, dev_uas))
            '''
            print("Test:")
            f.write("Test:")
            
            test_loss, test_uas = self.evaluate(test_loader)
            print("Loss:{}  UAS:{}".format(test_loss, test_uas))
            f.write("Loss:{}  UAS:{}\n".format(test_loss,test_uas ))
            '''
            if (max_dev < dev_uas):
                max_dev = dev_uas
                max_dev_epoch = epoch
            '''
            if (max_test < test_uas):
                max_test = test_uas
                max_test_epoch = epoch
            '''
            t = datetime.now() - start

            print("{}s elapsed".format(t))
            f.write("{}s elapsed\n".format(t))

            f.flush()
            total_time = total_time + t

        print("The max dev uas  is {} at {}".format(max_dev, max_dev_epoch))
        f.write("The max dev uas is {} at {}\n".format(max_dev, max_dev_epoch))

       # print("The max test uas is {} at {}".format(max_test, max_test_epoch))
       # f.write("The max test uas is {} at {}\n".format(max_test, max_test_epoch))

        print("The total time taken is {}".format(total_time))
        f.write("The total time taken is {}\n".format(total_time))

        print("The average time taken is {}".format(total_time / args.epochs))
        f.wrint("The average time taken is {}\n".format(total_time / args.epochs))
        f.close()










