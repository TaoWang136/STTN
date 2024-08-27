import os
import sys
import argparse
import numpy as np
from datetime import datetime
from sklearn import metrics
sys.path.append('../../')
from sttn_git.dataloader.data_process import risk_data
from sttn_git.models.model import STTN
from sttn_git.utils.lr_scheduler import LR_Scheduler
import sklearn.metrics as metrics
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
torch.manual_seed(22)


def getparse():
    parse = argparse.ArgumentParser()
    parse.add_argument('-SSM',type=str,default='MTTC')
    parse.add_argument('-pr_model_d',type=int,default=256)
    parse.add_argument('-s_model_d',type=int,default=64)

    parse.add_argument('-k',type=int,default=12)
    parse.add_argument('-spatial',type=str,help='choose the spatial model type')
    parse.add_argument('-pr',action='store_true')
    parse.add_argument('-s',action='store_true')
    parse.add_argument('-ST',action='store_true')
    parse.add_argument('-crash',type=int,choices=[0,1],default=0,help='in--0,out--1')
    parse.add_argument('-pr_t',type=str,default='p',choices=['t','p','tp','pr','r'])
    parse.add_argument('-s_t',type=str,default='c',choices=['t','p','tp','pr','r'])
    parse.add_argument('-traffic', type=str, default='crash')
    parse.add_argument('-proximal_size', type=int, default=3)#
    parse.add_argument('-period_size', type=int, default=2)#
    parse.add_argument('-test_size', type=int, default=60)
    parse.add_argument('-p_model_d',type=int,default=128)#
    parse.add_argument('-w',type=str)
    parse.add_argument('-save_dir', type=str, default='results')
    parse.add_argument('-best_valid_loss',type=float,default=1)
    parse.add_argument('-lr-scheduler', type=str, default='poly',choices=['poly', 'step', 'cos'])
    parse.add_argument('-warmup',type=int,default=100)
    parse.add_argument('-test_batch_size',type=int,default=1)
    parse.add_argument('-train', dest='train', action='store_true')
    parse.add_argument('-no-train', dest='train', action='store_false')
    parse.set_defaults(train=True)
#     parse.add_argument('-loss', type=str, default='l1', help='l1 | l2')
    parse.add_argument('-lr', type=float)
    parse.add_argument('-batch_size', type=int, default=16, help='batch size')
    parse.add_argument('-se',type=int)
    parse.add_argument('-epoch_size', type=int, default=50, help='epochs')
    parse.add_argument('-g',type=str,default=None)
    parse.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
       
    return parse.parse_args()


para = getparse()
print(para)
para.save_dir = '{}/{}'.format(para.save_dir, para.traffic)
if not os.path.exists(para.save_dir):
    os.makedirs(para.save_dir)
para.model_filename = '{}/proximal={}-period={}-spatial={}-pr={}-s={}-ST={}-scptmodel_d={}-{}-{}-{}'.format(
                    para.save_dir,para.proximal_size,para.period_size,para.spatial,para.pr,para.s,para.ST,
                    para.s_model_d,para.pr_model_d,para.p_model_d,'64')


se = 0
best_model = para.model_filename+'.model'
print(best_model)
if os.path.exists(best_model):
    saved = torch.load(best_model)
    se = saved['epoch']+1
    para.best_valid_loss = saved['valid_loss'][-1]
    lr = saved['lr']

lr = 0.001
if para.lr is not None:
    lr = para.lr
if para.se is not None:
    se = para.se
total_epochs = se + para.epoch_size




def getmetrics(pred,truth):
    print('get metrics with test data...')
    mae = metrics.mean_absolute_error(truth,pred)
    mse = metrics.mean_squared_error(truth,pred)
    rmse = mse**0.5
    nrmse = rmse/np.mean(truth)
    r2 = metrics.r2_score(truth,pred)
    return mae,mse,rmse,nrmse,r2 

def train_epoch(epoch):
    total_loss = 0
    model.train()
    data = train_loader

    i = 0
    for idx, (pr, p, target) in enumerate(data):

        scheduler(optimizer,i,epoch)
        optimizer.zero_grad()#清空过往梯度
        model.zero_grad()# 
        pred = model(pr.float(),para.pr,para.s,para.ST,para.pr_t,para.s_t,para.crash,p.float())
        loss = criterion(pred.float(), target.cuda().float()[:,:,para.crash])#计算预测值与实际标签之间的梯度
        #print(loss)
        total_loss += loss.item()
        loss.backward()#反向传播，计算当前梯度
        optimizer.step()#根据梯度更新网络参数
        i += 1
    return total_loss/len(data)

def valid_epoch(epoch):
    total_loss = 0
    model.eval()
    data = valid_loader
    i = 0
    for idx, (pr, p, target) in enumerate(data):
        optimizer.zero_grad()#清空过往梯度
        model.zero_grad()# 
        pred = model(pr.float(),para.pr,para.s,para.ST,para.pr_t,para.s_t,para.crash,p.float())
        loss = criterion(pred.float(), target.cuda().float()[:,:,para.crash])#计算预测值与实际标签之间的梯度
        #print(loss)
        total_loss += loss.item()
        loss.backward()#反向传播，计算当前梯度
        optimizer.step()#根据梯度更新网络参数
        i += 1
    return total_loss/len(data)





def train():
    best_valid_loss = para.best_valid_loss
    train_loss, valid_loss = [], []
    for i in range(se,total_epochs):
        train_loss.append(train_epoch(i))
        valid_loss.append(valid_epoch(i))
        if valid_loss[-1] < best_valid_loss:
            best_valid_loss = valid_loss[-1]
            torch.save({'epoch': i, 'model': model, 'train_loss': train_loss,
                        'valid_loss': valid_loss,'lr':optimizer.param_groups[0]['lr']}, para.model_filename + '.model')


def infer(test_type='train'):
    inference = []
    ground_truth = []
    test = []
    loss = []
    best_model = torch.load(para.model_filename + '.model').get('model')

    if test_type == 'train':
        data = train_loader
    elif test_type == 'test':
        data = test_loader
    elif test_type == 'valid':
        data = valid_loader

    for idx, (pr, p, target) in enumerate(data):
        pred = torch.relu(best_model(pr.float(),para.pr,para.s,para.ST,para.pr_t,para.s_t,para.crash,p.float()))
        inference.append(pred.float().data.cpu().numpy())
        ground_truth.append(target.float().numpy()[:,:,0])
        loss.append(criterion(pred.float(), target.cuda()[:,:,0]).float().item())


    final_predict = np.concatenate(inference)
    ground_truth = np.concatenate(ground_truth)
    truth = mmn.inverse_transform(ground_truth)
    predict = mmn.inverse_transform(final_predict)
    mae,mse,rmse,nrmse,r2 = getmetrics(predict.ravel(),truth.ravel())
    string =' [Real MSE]:{:0.5f}, [Real RMSE]:{:0.5f}, [Real NRMSE]: {:0.5f}, [Real MAE]:{:0.5f}, [Real R2]: {:0.5f}'.format(mse,rmse,nrmse,mae,r2)
    print(string)
 

def train_valid_split(dataloader, test_size=0.2, shuffle=True, random_seed=0):
    length = len(list(dataloader))
    indices = list(range(0, length))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    if type(test_size) is float:
        split = int(np.floor(test_size * length))
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('%s should be an int or float'.format(str))
    return indices[split:], indices[:split]


if __name__ == '__main__':
    riskdata= risk_data(para.proximal_size, para.period_size,para.test_size,para.SSM)#proximal_size, period_size, len_test, SSM):
    x_train, y_train, x_test, y_test, mmn =riskdata.load_and_process()
                                               
    x_train.append(y_train)
    x_test.append(y_test)
    train_data = list(zip(*x_train))
    test_data = list(zip(*x_test))
    # split the training data into train and validation
    train_idx, valid_idx = train_valid_split(train_data,0.1)
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
 
    train_loader = DataLoader(train_data, batch_size=para.batch_size, sampler=train_sampler,#num_workers=8,
                               pin_memory=True,drop_last=True)
    valid_loader = DataLoader(train_data, batch_size=para.batch_size, sampler=valid_sampler,#num_workers=8,
                               pin_memory=True,drop_last=True)

    test_loader = DataLoader(test_data, batch_size=para.test_batch_size, shuffle=False,drop_last=True)

    external_size = 6,print('train_data.shape',len(train_data))
    if para.g is not None:
        GPU = par.g
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    print("preparing gpu...")
    if torch.cuda.is_available():
        print('using Cuda devices, num:',torch.cuda.device_count())
        print('using GPU:',torch.cuda.current_device())

    if os.path.isfile(best_model):
        #print(best_model)
        model = torch.load(best_model)['model'].cuda()
    else:
        model = STTN(para.proximal_size, external_size, para.k, para.spatial,para.pr_model_d,para.s_model_d,para.p_model_d).cuda()#加载transformer模型，.cuda()使其在GPU中运算。
    scheduler = LR_Scheduler(para.lr_scheduler, lr, total_epochs, len(train_loader),warmup_epochs=para.warmup)#定义学习率
    optimizer = optim.Adam(model.parameters(),lr,betas=(0.9, 0.98), eps=1e-9)


    criterion = nn.MSELoss().cuda()
    print('Training...')
    if para.train:
        train()
    #predict('train')
    infer('test')
