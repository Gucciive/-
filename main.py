# -*- coding: utf-8 -*-
import json
import logging
import os
from multiprocessing import Queue

import numpy as np
import paddle
from paddle import nn
import multiprocessing

class DecoderLayer(nn.Layer):
    def __init__(self,d_model,nhead,dim_feedforward,dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiHeadAttention(
            d_model,
            nhead,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward )
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
    def forward(self,x,attn_mask):
        r=x
        x=self.self_attn(x,x,x,attn_mask=attn_mask)
        x=self.dropout(x)
        x=x+r
        x=self.norm1(x)
        r=x
        x =self.linear1(x)
        x=self.activation(x)
        x=self.dropout1(x)
        x=self.linear2(x)
        x=self.dropout2(x)
        x=x+r
        x=self.norm2(x)
        # x = self.norm1(x+self.dropout(self.self_attn(x,x,x,attn_mask=attn_mask,)))
        # x = self.norm2(x + self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x))))))
        return x
class WYY (nn.Layer):
    def __init__(self,d_model,vocab_size,nlayer,nhead,max_len,batch_size,dim_feedforward,pad_id):
        super(WYY, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.position=paddle.concat([paddle.arange(0,max_len).unsqueeze(0) for _ in range(batch_size)],axis=0)
        self.Decoders= self.decoder_block(nlayer, d_model, nhead, dim_feedforward)
        # 输出线性层
        self.fc_out = nn.Linear(d_model, vocab_size)
        # self.max_len=max_len
        self.atten_mask= paddle.tril(paddle.ones(shape=(max_len, max_len), dtype=paddle.bool)).unsqueeze(0)#tril
        # self.atten_mask=paddle.triu(paddle.ones(shape=(max_len, max_len), dtype=paddle.bool),diagonal=1).unsqueeze(0)#triu
        # print(self.atten_mask)
        # exit()
        self.atten_mask=paddle.concat([self.atten_mask for _ in range(nhead)],axis=0).unsqueeze(0)
        self.atten_mask=paddle.concat([self.atten_mask for _ in range(batch_size)],axis=0)
    def decoder_block(self,n,d_model,nhead,dim_feedforward):
        return nn.LayerList([DecoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward) for _ in range(n)])
    def forward(self, x):
        p=self.position_embedding(self.position)
        x=self.embedding(x)
        x=x+p
        for Decoder in self.Decoders:
            x=Decoder(x,self.atten_mask)
        x=self.fc_out(x)
        return x
    def save_model(self):
        paddle.save(self.state_dict(), "adam.pdparams")
class Dict(object):
    def __init__(self):
        self.vocab_path= 'vocab.json'

        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

    def vocab_to_id(self,data,max_length):
        datas=[]
        temp=[self.vocab['<s/>']]
        for words in data:
            for word in words:
                    temp.append(self.vocab.get(word,self.vocab['<unk>']))

            temp.append(self.vocab['<end>'])
            temp.append(self.vocab['<s/>'])
        temp.pop(-1)
        d= self.pad_or_crop(temp,max_length)
        for i in d:
            datas.append(i)
        return datas
    def pad_or_crop(self,datas,max_length):
        d=[]
        b=0
        e=max_length
        while True:
            if e>len(datas):
                temp=datas[b:]
                temp.extend([self.vocab["<pad>"]] * (max_length - (len(datas) - b)))
                d.append(temp)
                break
            elif e<len(datas):
                d.append(datas[b:e])
            else:
                d.append(datas[b:e])
            b+=max_length
            e+=max_length
        return d
class DataDeal(object):
    def __init__(self,max_len):
        self.dict = Dict()
        self.max_len=max_len
    def data_deal(self,lines,mode="json"):
        datas=[]
        data=[]
        if mode=="json":
            for line in lines:
                data.append( json.loads(line)['content'])

            data_id= self.dict.vocab_to_id(data,self.max_len)
            for i in range(len(data_id)):
                datas.append(data_id[i])
        else:
            data="".join(lines)
            data = self.dict.vocab_to_id(data, self.max_len)
            for i in range(len(data)):
                datas.append(data[i])
        return datas
class Dataset(paddle.io.Dataset):
    def __init__(self, data, max_len,data_deal):
        self.datas = data_deal.data_deal(data)
        self.datas=np.array(self.datas)
        self.max_len = max_len
    def __getitem__(self, index):
        return self.datas[index]
    def __len__(self):
        return len(self.datas)
class DatasetQueue(object):
    def __init__(self, path, max_len,max_queue_size=10):
        self.queue = Queue(maxsize=max_queue_size)
        self.path = path
        self.max_len = max_len
        self.max_queue_size=max_queue_size
    def put(self,queue):
        paths = os.listdir(self.path)
        paths = [os.path.join(self.path, path) for path in paths]
        datas = []
        data_deal = DataDeal(self.max_len)
        for path in paths:
            end = False
            with open(path, 'r', encoding='utf-8') as f:
                while True:
                    for _ in range(1000):
                        data = f.readline()
                        if not data:
                            end = True
                            break
                        datas.append(data)
                    dataset = Dataset(datas, max_len=self.max_len, data_deal=data_deal)
                    queue.put(dataset)
                    datas = []
                    if end:
                        break



class DataLoaders(object):
    def __init__(self, batch_size, dataset_queue,shuffle=True,drop_last=True):
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.drop_last=drop_last
        self.dataset_queue=dataset_queue

    def get_dataloader(self):
        while True:
            dataset=self.dataset_queue.get()
            dataloader = paddle.io.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                              drop_last=self.drop_last)
            yield dataloader


def save_checkpoint(model,optimizer,scheduler,epoch,data_num,current_clip_norm):
    state = {
        'epoch': epoch,
        'data_num': data_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),  # 保存学习率调度器的状态
        'current_clip_norm': current_clip_norm
    }
    paddle.save(state, 'checkpoint.pth')
class Logger(object):
    def __init__(self):
        # 创建 Logger
        self.logger=logging.getLogger('my_logger')
        self.logger.setLevel(logging.DEBUG)
        # 创建 Console Handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)

        # 创建 File Handler
        self.file_handler = logging.FileHandler('app.log',encoding='utf-8')
        self.file_handler.setLevel(logging.DEBUG)

        # 创建 Formatter
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # 将 Formatter 添加到 Handler
        self.console_handler.setFormatter(self.formatter)
        self.file_handler.setFormatter(self.formatter)

        # 将 Handler 添加到 Logger
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.file_handler)
    def debug(self,message):
        self.logger.debug(message)
    def info(self,message):
        self.logger.info(message)
    def warning(self,message):
        self.logger.warning(message)
    def error(self,message):
        self.logger.error(message)
    def critical(self,message):
        self.logger.critical(message)

def train(path,nlayer,nhead,total_steps,vocal_size,d_model,max_len,batch_size,target_batch_size,epoch,dim_feedforward,pad_id,max_queue_size,device, weight_decay=None, clip_norm=1.0):
    logger=Logger()
    model=WYY(vocab_size=vocal_size,nlayer=nlayer,nhead=nhead,d_model=d_model,max_len=max_len,batch_size=batch_size,dim_feedforward=dim_feedforward,pad_id=pad_id)
    vocab_weight =paddle.load("vocab_weight.pdtensor")
    loss=nn.CrossEntropyLoss(reduction='mean',ignore_index=pad_id,label_smoothing=0.3,weight=vocab_weight,)
    scheduler = paddle.optimizer.lr.OneCycleLR(
        max_learning_rate=0.0001,  # 最大学习率
        total_steps=total_steps,  # 总的训练步数
        end_learning_rate=0.000001
    )
    initial_clip_norm = clip_norm
    current_clip_norm = initial_clip_norm
    grad_clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=current_clip_norm)
    # 监控梯度范数
    gradient_norms = []
    optimizer=paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=scheduler,weight_decay=weight_decay,grad_clip=grad_clip)
    data_num=0
    start_epoch=0
    if os.path.exists("checkpoint.pth"):
        checkpoint = paddle.load("checkpoint.pth")
        model.set_state_dict(checkpoint['model_state_dict'])
        optimizer.set_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.set_state_dict(checkpoint['scheduler_state_dict'])  # 加载学习率调度器的状态
        start_epoch = checkpoint['epoch']
        data_num = checkpoint['data_num']
        current_clip_norm = checkpoint['current_clip_norm']
        grad_clip.clip_norm = current_clip_norm

    model.train()
    model.to(device)
    for epoch_ in range(start_epoch,epoch):
            logger.info("开始第{}轮训练".format(epoch_))
            dataset_queue=DatasetQueue(path=path,max_len=max_len,max_queue_size=max_queue_size)
            multiprocessing.Process(target=dataset_queue.put,args=(dataset_queue.queue,)).start()
            dataloaders=DataLoaders(batch_size=batch_size,dataset_queue=dataset_queue.queue)
            logger.info("成功创建数据集生成器")
            get_dataloader=dataloaders.get_dataloader()
            optimizer.clear_grad()
            for _ in range(data_num):
                next(get_dataloader)
            for dataloader in get_dataloader:
                grad_num=0
                for data in dataloader():
                    data=paddle.to_tensor(data,place=device)
                    r = model(data)
                    r = r[:, 1:-1,:]
                    r=r.reshape((-1,r.shape[-1]))
                    y=data[:,2:].reshape((-1,))
                    # print(y[:100])
                    l = loss(r, y)
                    l.backward()
                    # logger.info("完成损失计算损失为{}".format(l.item()))
                    grad_num+=1
                    if  grad_num % (target_batch_size // batch_size) == 0:
                        global_norm =np.sqrt(sum([np.sum(np.square(p.grad.numpy())) for p in model.parameters()]))
                        gradient_norms.append(global_norm)
                        paddle.set_printoptions(threshold=100000)
                        print(paddle.topk(r,k=3)[1])
                        optimizer.step()
                        scheduler.step()
                        optimizer.clear_grad()
                        if len(gradient_norms) > 100:  # 每100个batch检查一次
                            avg_norm = np.mean(gradient_norms[-100:])
                            if avg_norm > current_clip_norm * 1.5:
                                current_clip_norm *= 1.1  # 增加 clip_norm
                                grad_clip.clip_norm = current_clip_norm
                            elif avg_norm < current_clip_norm * 0.5:
                                current_clip_norm /= 1.1  # 减少 clip_norm
                                grad_clip.clip_norm = current_clip_norm
                            gradient_norms = gradient_norms[10:]
                            print("------",current_clip_norm)
                data_num+=1
                save_checkpoint(model,optimizer,scheduler,epoch_,data_num,current_clip_norm)
            data_num=0
def predict(nlayer,nhead,vocal_size,d_model,max_len,batch_size,dim_feedforward,pad_id,device):
    model=WYY(vocab_size=vocal_size,nlayer=nlayer,nhead=nhead,d_model=d_model,max_len=max_len,batch_size=batch_size,dim_feedforward=dim_feedforward,pad_id=pad_id)
    if os.path.exists("checkpoint.pth"):
        # print(302)
        # exit()
        checkpoint = paddle.load("checkpoint.pth")
        model.set_state_dict(checkpoint['model_state_dict'])
    model.train()
    model.to(device)
    text='快科技12月31日消息，今日，有博主发文称，有个朋友只是和吴柳芳同名，'# print(text,end="")
    with open("vocab1.json", 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    id_to_vocab = {v: k for k, v in vocab.items()}
    data=[vocab["<s/>"]]
    for word in text:
        data.append(vocab.get(word, vocab["<unk>"]))
    l=len(data)
    # print(l)
    # exit()
    if l<max_len:
        data.extend([vocab["<pad>"]]*(max_len-len(data)))
    elif l>max_len:
        return False
    data=[data]

    data=paddle.to_tensor(data,place=device)
    for _ in range(10):
        output=model(data)[0,l-1,:]
        paddle.set_printoptions(precision=4,threshold=8000)
        # print(paddle.topk(output, 3)[1])
        # exit()
        output=paddle.argmax(output)
        print(id_to_vocab[output.item()],end="")
        data[0][l]=output
        # print(data)
        l+=1




if __name__ == '__main__':
    # predict(vocal_size=6463,d_model=768,max_len=512,batch_size=1,dim_feedforward=3072,pad_id=0,nlayer=12,nhead=12,device=paddle.device.get_device())
    train(path="D:\code\datas\\1623_0000001\zh",total_steps=125492*5,vocal_size=4215,d_model=768,max_len=512,batch_size=8,target_batch_size=80,epoch=5,dim_feedforward=3072,pad_id=0,max_queue_size=100,nlayer=12,nhead=12,device=paddle.device.get_device())
