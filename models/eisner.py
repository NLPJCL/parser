import time
import sys


class Eisner():
    def __init__(self):
        #特征权重，特征到权重的映射str->float
        self.w={}
    def creat_feature(self,sentence,tags,i,j):
        #i为修饰词，j为核心词。
        feas=[]
        #一元特征
        feas.append(sentence[j]+tags[j])#p-word,p-pos
        feas.append(sentence[j])
        feas.append(tags[j])

        feas.append(sentence[i] + tags[i])  # c-word,c-pos
        feas.append(sentence[i])
        feas.append(tags[i])
        #二元特征
        feas.append(sentence[j] + tags[j]+sentence[i]+tags[i])  # c-word,c-pos
        feas.append(tags[j]+sentence[i]+tags[i])

        feas.append(sentence[j]+sentence[i]+tags[i])
        feas.append(sentence[j]+tags[j]+tags[i])
        feas.append(sentence[j]+tags[j]+sentence[i])
        feas.append(sentence[j]+sentence[i])
        feas.append(tags[j]+tags[i])
        #依存弧间的单词的词性特征
        i_j_tags=[]
        if i<j:
            for q in range(i,j+1):
                i_j_tags.append(tags[q])
        elif j<i:
            for q in range(j,i+1):
                i_j_tags.append(tags[q])
        feas.append(''.join(i_j_tags))
        #周围的单词的词性特征
        #分情况讨论核心词在最后一个或不在最后一个的情况
        #修饰词在最后一个时，它的下一个用%%
        #核心词在最后一个时，它的下一个用##

        if (j<len(sentence)-1):
            feas.append(tags[j]+tags[j+1]+tags[i-1]+tags[i])#j+1            #1
            if(i<len(sentence)-1):
                feas.append(tags[j]+tags[j+1]+tags[i]+tags[i+1])#j+1            #3
            else:
                feas.append(tags[j]+tags[j+1]+tags[i]+'%%')#j+1            #3
        else:#若核心词是最后一个，那么tags[j+1]用##代替
            feas.append(tags[j]+'##'+tags[i-1]+tags[i])#j+1                 #1
            if(i<len(sentence)-1):
                feas.append(tags[j]+'##'+tags[i]+tags[i+1])#j+1                  #3
            else:
                feas.append(tags[j]+'##'+tags[i]+'%%')#j+1                  #3

        #分情况讨论核心词在第一个，或不在第一个的情况
        # 若核心词是第一个,那么tags[j-1]用$$代替
        #
        if (j==0):#核心词在第一个
            feas.append('$$' + tags[j] + tags[i - 1] + tags[i])  # j-1          #2
            # 分情况讨论修饰词在最后一个，或不在最后一个的情况
            if (i<len(sentence)-1):
                feas.append('$$' + tags[j] + tags[i] + tags[i + 1])  # j-1,i+1  #4
            else:#如果修饰词是最后一个那么tags[i+1]用%%代替
                feas.append('$$' + tags[j] + tags[i] + '%%')  # j-1,i+1         #4
        else:
            feas.append(tags[j-1]+tags[j]+tags[i-1]+tags[i])   #j-1#2         #2
            # 分情况讨论修饰词在第一个，或不在第一个的情况
            if (i<len(sentence)-1):
                feas.append(tags[j-1]+tags[j]+tags[i]+tags[i+1])#j-1,i+1      #4
            else:
                feas.append(tags[j-1]+tags[j]+tags[i]+'%%')#j-1,i+1           #4
        return feas

    def creat_feature_space(self,train_dataset):
    #train_dataset=[(句子,词性，(i,核心词，标签))]
        for q in range(len(train_dataset)):
            for i in range(1,len(train_dataset[q][0])):#修饰词 因为修饰词不可能是$0。因此从1开始
                for j in range(len(train_dataset[q][0])):#核心词 因为核心词可能是$0，因此从0开始
                    if(j==i):#一个词不可能即是修饰词又是核心词。
                        continue
                    #feas=self.creat_feature(train_dataset[q][0],train_dataset[q][1],i,j)
                    for fea in self.creat_feature(train_dataset[q][0],train_dataset[q][1],i,j):
                        if fea in self.w:
                            pass
                        else:
                            self.w[fea]=0
        print("特征空间的大小为%d"%(len(self.w)))
    def count_score(self,feas):
        total_score=0
        for fea in feas:
            if fea in self.w:
                total_score+=self.w[fea]
        return total_score
    def eisner(self,sentence,tags):
        #I_f,I_b表示不完整的span,从前向后和从后向前.C_f,C_b同理。

        sentence_length=len(sentence)
        I=[[0]*sentence_length  for i in range(sentence_length)]#矩阵对角线公用，上半部分装i->j，下半部分装j->i。
        C=[[0]*sentence_length  for i in range(sentence_length)]

        I_path=[[0]*sentence_length  for i in range(sentence_length)]
        C_path=[[0]*sentence_length  for i in range(sentence_length)]

        head=[]#来计算头节点对应的索引。

        for w in range(1,sentence_length):
            for i in range(0,sentence_length-w):
                j=i+w
                #选择从i到j的不完整的span。
                I_path[i][j] = i
                for r in range(i,j):
                    feas=self.creat_feature(sentence,tags,j,i)#creat_feature是修饰词到核心词。
                    score=C[i][r]+C[j][r+1]+self.count_score(feas)
                    if score>=I[i][j]:
                        I[i][j]=score
                        I_path[i][j]=r

                #选择从j到i的不完整的span。
                I_path[j][i] = r
                for r in range(i,j):
                    feas=self.creat_feature(sentence,tags,i,j)
                    score=C[i][r]+C[j][r+1]+self.count_score(feas)
                    if score>=I[j][i]:
                        I[j][i]=score
                        I_path[j][i]=r


                #选择从i到j的完整的span。
                C_path[i][j] = i+1
                for r in range(i+1, j+1):
                    score = I[i][r] + C[r][j]
                    if score >=C[i][j]:
                        C[i][j] = score
                        C_path[i][j]=r

                #选择从j到i的完整的span。
                C_path[j][i] = i
                for r in range(i, j):
                    score = C[r][i] + I[j][r]
                    if score >=C[j][i]:
                        C[j][i] = score
                        C_path[j][i]=r

        heads=[-1]*(sentence_length)#索引为修饰词，值对应为核心词。
        self.backtrack(I_path,C_path,heads,0,sentence_length-1,True)
        return heads
    def backtrack(self,I_path,C_path,heads,i,j,complete):
        if i==j:
            return
        if complete:
                r = C_path[i][j]
                self.backtrack(I_path,C_path,heads,i,r,False)#也是大的到小的，所有i到r。
                self.backtrack(I_path,C_path,heads,r,j,True)
        else:
            heads[j]=i
            r=I_path[i][j]
            i,j=sorted((i,j))
            self.backtrack(I_path, C_path, heads, i, r, True)
            self.backtrack(I_path, C_path, heads, j,r+1,True)

    def updata(self,sentence,tags,glod_tree,max_tree):
        #如果评测出来的依存树不等于正确的依存树,那么把正确的依存树的弧的特征向量加1，错误的减1.
        if glod_tree!=max_tree:
            #把正确的对应的特征向量加1，错误的减1.
            for i in range(1,len(glod_tree)):
                corr_feas=self.creat_feature(sentence,tags,i,glod_tree[i])
                for fea in corr_feas:
                    if fea in self.w:
                        self.w[fea]+=1
                err_feas=self.creat_feature(sentence,tags,i,max_tree[i])
                for fea in err_feas:
                    if fea in self.w:
                        self.w[fea]-=1
    def evaluate(self,dataset):
        total_word=0
        corr_head=0
        for i in range(len(dataset)):
            glod_tree=dataset[i][2]
            max_tree=self.eisner(dataset[i][0],dataset[i][1])
            if len(glod_tree)!=len(max_tree):
                print("句子长度不一样")
            for j in range(len(max_tree)):
                if max_tree[j]==glod_tree[j]:
                    corr_head+=1
            total_word+=len(glod_tree)
        return corr_head/total_word

    def online_training(self,iterator,shuffle,train_dataset,dev_dataset,test_dataset):
        for i in range(1,iterator+1):
            start = time.time()
            for j in range(len(train_dataset)):
                if j%100==0:
                    print(j,len(train_dataset[j][0]))
                glod_tree=train_dataset[j][2]
                max_tree=self.eisner(train_dataset[j][0],train_dataset[j][1])
                self.updata(train_dataset[j][0],train_dataset[j][1],glod_tree,max_tree)
            #评估
            train_UAS=self.evaluate(train_dataset)
            print("Train UAS:%f"%(train_UAS))

            dev_UAS=self.evaluate(dev_dataset)
            print("Dev UAS:%f"%(dev_UAS))

            test_UAS=self.evaluate(test_dataset)
            print("Test UAS:%f"%(test_UAS))
            end = time.time()
            print(end - start)



