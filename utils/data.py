class Data():
    def __init__(self):
        pass
    def load(self,file_name):
        sentence_num=0
        with open(file_name,'r',encoding='utf-8') as f:
            dataset=[]
            sentence=["$0"]
            tags=["$0"]
            arcs=[-1]
            line=f.readline()
            while(line!=''):
                if line=='\n':
                    dataset.append((sentence,tags,arcs))
                    sentence = ["$0"]
                    tags = ["$0"]
                    arcs = [-1]
                    line=f.readline()
                    sentence_num+=1
                    continue
                line_lst=line.split()
                sentence.append(line_lst[1])
                tags.append(line_lst[3])
                #修饰词索引，核心词索引,对应关系、
                #arcs.append((int(line_lst[0]),int(line_lst[6]),line_lst[7]))
                arcs.append(int(line_lst[6]))
                line=f.readline()
            dataset.append((sentence, tags, arcs))

            print("%s有%d句子"%(file_name,sentence_num))
            return dataset


