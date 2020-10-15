import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torch.nn.utils.rnn import pad_sequence

class Net_Data():
    def __init__(self,config,fix_len=20):
        self.config=config
        self.wdict = {}  # word to id
        self.tag={} # tag to id
        self.label={}
        self.char={}
        # 加载词表和标签表
        self.load_word_tag_label(config.ftrain)
        self.load_embedding(config.fembed,config.n_embed,config.n_feat_embed)
        self.fix_len=fix_len
        if config.feat=='char':
            self.n_feats=len(self.char)
        else:
            self.n_feats=len(self.tag)



       # self.label_num = len(self.label)

    def load_embedding(self,embedding_file,n_embed,n_tag_embed):
        embed_indexs=[]
        embed=[]
        with open(embedding_file,'r',encoding='utf-8') as f:
            lines=f.readlines()
            for i in range(len(lines)):#从1开始，读取出词嵌入
                word_value=lines[i].split(' ')
                if word_value[0] in self.wdict:#如果当前词在wdict中，那么把其索引加入。
                    embed_indexs.append(self.wdict[word_value[0]])
                else:#如果没在，先添加其索引到embed_index中，然后再加入。(不能反过来)
                    embed_indexs.append(len(self.wdict))
                    self.wdict[word_value[0]]=len(self.wdict)
                embed.append(list(map(float,word_value[1:])))
        embed=torch.tensor(embed,dtype=torch.float)
        #创建embed，然后随机初始化。
        self.word_embed=torch.Tensor(len(self.wdict),n_embed)
        std = (1.0/ self.word_embed.size(1)) ** 0.5
        nn.init.normal_(self.word_embed, mean=0, std=std)
        #给词向量矩阵赋值、
        self.word_embed[embed_indexs] = embed

        #给词性向量赋值
        self.tag_embed=torch.Tensor(len(self.tag),n_tag_embed)
        std = (1.0/ self.tag_embed.size(1)) ** 0.5
        nn.init.normal_(self.tag_embed, mean=0, std=std)

        print("载入词嵌入文件后共有%d个词"%(len(self.wdict)))

    #加载文件
    def read_file(self,filename):
        sentences, tags,chars,arcs,labels= [], [],[],[],[]
        sentence,tag,char,arc,label=[],[],[],[],[]
        with open(filename, 'r', encoding='utf-8') as fin:
            for line in fin:
                if line=='\n':
                    sentences.append(sentence)
                    tags.append(tag)
                    chars.append(char)
                    arcs.append(arc)
                    labels.append(label)
                    
                    sentence=[]
                    char=[]
                    tag=[]
                    arc=[]
                    label=[]
                    continue
                line_lst = line.strip().split()
                sentence.append(line_lst[1])  # 分词成列表
                tag.append(line_lst[3])
                char.append([char_ for char_ in line_lst[1]])
                arc.append(int(line_lst[6]))#给其把str转换为int
                label.append(line_lst[7])

        print("%s有%d个句子"%(filename,len(sentences)))
        return sentences, tags,chars,arcs,labels

    #生成词表
    def load_word_tag_label(self,train_file):
        sentences,tags,chars,_,labels=self.read_file(train_file)
        # 添加未知词和空格词
        self.wdict['<PAD>']=len(self.wdict)
        self.wdict['<BLANK>']=len(self.wdict)
        self.wdict['<UNK>']=len(self.wdict)
        self.wdict['<BOS>']=len(self.wdict)

        self.tag['<PAD>']=len(self.tag)
        self.tag['<BOS>']=len(self.tag)

        self.label['<PAD>']=len(self.label)
        self.label['<BOS>']=len(self.label)

        self.char['<PAD>']=len(self.char)
        self.char['<UNK>']=len(self.char)
        self.char['<BOS>']=len(self.char)

        #加载词和词性
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                if sentences[i][j] not in self.wdict:
                    self.wdict[sentences[i][j]]=len(self.wdict)
                if tags[i][j] not in self.tag:
                    self.tag[tags[i][j]]=len(self.tag)
                if labels[i][j] not in self.label:
                    self.label[labels[i][j]]=len(self.label)

                for k in range(len(chars[i][j])):
                    if chars[i][j][k] not in self.char:
                        self.char[chars[i][j][k]]=len(self.char)

        print("train中共有%d个词"%(len(self.wdict)))
        print("train中共有%d个标签"%(len(self.tag)))
        print('train中共有%d个字'%(len(self.char)))

    def load_file(self,file_name):
        sentences, tags,chars,arcs, labels=self.read_file(file_name)
        all_sen=[]
        all_tag=[]
        all_char=[]
        all_arc=[]
        all_label=[]
        lens=[]
        for sen,tag,char,arc,label in zip(sentences,tags,chars,arcs,labels):
            x,y=[self.wdict['<BOS>']],[self.tag['<BOS>']]
            z=[[self.char['<BOS>'] for q in range(self.fix_len)]]
            new_arc=[len(sen)+1]+arc
            for i in range(len(sen)):
                if sen[i] in self.wdict:
                    x.append(self.wdict[sen[i]])
                else:
                    x.append(self.wdict['<UNK>'])
                y.append(self.tag[tag[i]])
                char_idx=[]
                for j in range(20):
                    if j<len(char[i]):
                        if char[i][j] in self.char:
                            char_idx.append(self.char[char[i][j]])
                        else:
                            char_idx.append(self.char['<UNK>'])
                    else:
                        char_idx.append(0)
                z.append(char_idx)

            all_sen.append(torch.tensor(x,dtype=torch.long))
            all_tag.append(torch.tensor(y,dtype=torch.long))
            all_char.append(torch.tensor(z,dtype=torch.long))

            all_arc.append(torch.tensor(new_arc,dtype=torch.long))
            label=[self.label['<BOS>']]+[self.label[lab] for lab in label]
            all_label.append(torch.tensor(label, dtype=torch.long))
            lens.append(len(y))

        all_sen=pad_sequence(all_sen,True,self.wdict['<PAD>'])
        all_tag=pad_sequence(all_tag,True,self.wdict['<PAD>'])
        all_char=pad_sequence(all_char,True,self.char['<PAD>'])
        
        all_arc=pad_sequence(all_arc,True,self.wdict['<PAD>'])
        all_label=pad_sequence(all_label,True,self.wdict['<PAD>'])
        lens=torch.tensor(lens)
        if self.config.feat=='tag':
            data=TensorDataset(all_sen,all_tag,all_arc,all_label,lens)
        else:
            data=TensorDataset(all_sen,all_char,all_arc,all_label,lens)
        return data
