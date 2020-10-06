import config
from  utils import Net_Data
from models import Tag_Biaffine,CRF_Biaffine
import argparse
from torch.utils.data import DataLoader
import torch
from datetime import datetime, timedelta

def collate_fn(data):
    # 对指定的batchsize数据按照长度大小进行排序。
    sen, tag,arc,label,lens = zip(
        *sorted(data, key=lambda x: x[-1], reverse=True)
    )
    max_len = lens[0]
    sen = torch.stack(sen)[:, :max_len]
    tag = torch.stack(tag)[:, :max_len]
    arc=torch.stack(arc)[:,:max_len]
    label=torch.stack(label)[:,:max_len]
    lens = torch.tensor(lens)

    return sen,tag,arc,label,lens
def collate_fn_cuda(data):
    # 对指定的batchsize数据按照长度大小进行排序。
    sen, tag,arc,label,lens = zip(
        *sorted(data, key=lambda x: x[-1], reverse=True)
    )
    max_len = lens[0]

    sen = torch.stack(sen)[:, :max_len].to(device)
    tag = torch.stack(tag)[:, :max_len].to(device)
    arc=torch.stack(arc)[:,:max_len].to(device)
    label=torch.stack(label)[:,:max_len].to(device)
    lens = torch.tensor(lens).to(device)

    return sen,tag,arc,label,lens


if __name__=='__main__':
    #神经网络模型
    parser = argparse.ArgumentParser(description="神经网络参数配置")
    parser.add_argument("--model", default='npcrf_biaffine',
                        choices=['global_eisner', 'tag_biaffine','npcrf_biaffine'],
                        help='choose the model for paser')
    args = parser.parse_args()

    # 设计随机数种子和线程
    torch.manual_seed(1)
    torch.set_num_threads(8)

    if (args.model=='tag_biaffine'):
        config = config.config["tag_biaffine"]
    else:
        print("np crf parser")
        config=config.config["npcrf_biaffine"]
    dataset=Net_Data(config.ftrain,config.fembed,config.n_embed,config.n_tag_embed)
    trainset=dataset.load_file(config.ftrain)
    devset=dataset.load_file(config.fdev)
    testset=dataset.load_file(config.ftest)

    if config.use_gpu==True:
            device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
            print(device)
    train_loader = DataLoader(dataset=trainset,
                             batch_size=config.batch_size,
                             shuffle=True,
                            collate_fn=collate_fn_cuda  if  config.use_gpu else collate_fn )

    dev_loader = DataLoader(dataset=devset,
                             batch_size=config.batch_size,
                             shuffle=True,
                             collate_fn=collate_fn_cuda  if  config.use_gpu else collate_fn)

    test_loader = DataLoader(dataset=testset,
                             batch_size=config.batch_size,
                             shuffle=True,
                             collate_fn=collate_fn_cuda  if  config.use_gpu else collate_fn)

    file_nema="np_partial_result_norm_unbias.txt"
    f=open(file_nema,'w')
    if (args.model=='tag_biaffine'):
        net=Tag_Biaffine(dataset.word_embed,dataset.tag_embed,config)
    else:
        print('npcrf')
        net=CRF_Biaffine(dataset.word_embed,dataset.tag_embed,config)


    if config.use_gpu==True:
        net.to(device)
    net.fit(config,train_loader,dev_loader,test_loader,f,device)
    f.close()

  #  parser= tag_biaffine(args)

    '''
    #global linear model
    #加载配置文件
    config = config.config["global_eisner"]
    #加载数据集
    dataset=Data()
    train_dataset=dataset.load(config.ftrain)
    dev_dataset=dataset.load(config.fdev)
    test_dataset=dataset.load(config.ftest)

    #加载模型
    global_eisner=Eisner()
    start=time.time()
    #创建特征空间
    global_eisner.creat_feature_space(train_dataset)
    end=time.time()
    print(end-start)
    #在线训练
    global_eisner.online_training(config.iterator,config.shuffle,train_dataset,dev_dataset,test_dataset)
    '''