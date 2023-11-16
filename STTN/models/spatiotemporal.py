from sttn_prediction.models.transformer import train_model
from sttn_prediction.models.utils import pr_subsequent_mask,corr
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math
from .utils import corr#getadj,get_adj

class proximal(nn.Module):
    def __init__(self,k,N,model_d):
        super(proximal,self).__init__()
        self.pr_temporal =train_model(k+1,1,N,model_d)
        self.k = k
    def forward(self,x_pr,x_p,tgt_mode,mode,number,adj=None,index=None,x_t=None):
        '''initial data size
        '''
        bs = len(x_pr)
        N = x_pr.shape[-1]
        len_proximal = x_pr.shape[1]
        #adj
        sx_pr = x_pr.permute((0,2,3,1)).float()
        if adj is None:
            adj = corr(sx_pr)
        if index is None: 
            index = torch.argsort(adj,dim=-1,descending=True)[:,:,0:self.k]
        selected = torch.zeros((bs,N,self.k,len_proximal),dtype=torch.float)
        for i in range(bs):
            for j in range(N):
                    selected[i,j] = sx_pr[i,number,index[i,j]]

        tx_pr = torch.cat([sx_pr[:,number].unsqueeze(-1),selected.transpose(-1,-2)],dim=-1).cuda()

        tgt_mask_pr = pr_subsequent_mask(len_proximal).cuda()
        if(tgt_mode=='pr'):
            tgt_c = sx_c[:,number].unsqueeze(-1).cuda()
        else:
            tgt_c= torch.mean(x_p[:,:,:,number],dim=1).transpose(1,2).unsqueeze(-1).cuda()
        sq_pr = self.pr_temporal(tx_pr, tgt_c, tgt_mask_pr).squeeze(-1)
        return sq_pr
    
   
class period(nn.Module):
    def __init__(self,proximal_size,N,model_d):
        super(period,self).__init__()
        self.p_temporal = train_model(proximal_size,proximal_size,N,model_d)
 
    def forward(self,x_pr,x_p,flow):

        bs = len(x_pr)
        N = x_pr.shape[-1]
        len_proximal = x_pr.shape[1]

        tgt = x_pr.permute((0,2,3,1))[:,flow].unsqueeze(dim=-2).cuda()
        tx_p= x_p.permute(0,3,4,1,2).float().cuda()

        sq_p = self.p_temporal(tx_p[:,flow], tgt).squeeze(dim=-2)
        return sq_p

class Spatial(nn.Module):
    def __init__(self,proximal_size,k,N,model_d):
        super(Spatial,self).__init__()
        self.spatial = train_model(proximal_size,proximal_size,N,model_d,spatial=True)
        self.k = k  
    def forward(self,x_pr,x_p,tgt_mode,mode,number,A=None,index=None,x_t=None):
        bs,proximal,_,N = x_pr.shape#1*3*2*400
        x = x_pr.permute((0,2,3,1)).float()#1*2*400*3
        if A is None:
            A = getA_corr(x)
        sx_pr= torch.zeros((bs,N,self.k,proximal),dtype=torch.float32)#1*400*20*3
        if index is None:
            index = torch.argsort(A,dim=-1,descending=True)[:,:,0:self.k] #bs,N,k
        for i in range(bs):
            for j in range(N):
                sx_pr[i,j] = x[i,number,index[i,j]]#1*400*20*3
            tgt = torch.mean(x_p[:,:,:,number],dim=1).transpose(1,2).unsqueeze(-2).cuda()
        sq_pr = self.spatial(sx_pr.cuda(),tgt).squeeze(dim=-2)#输入模型，一个是source sequence1*400*20*3，一个是initial sequence1*400*1*3
        return sq_pr,tgt.permute((0,2,3,1)).squeeze(1)#输出sq_c1*400*3和1*3*400
 