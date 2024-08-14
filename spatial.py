import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from STTN.models.transformer import make_model
from .utils import getA_cosin,getA_corr,getadj,get_adj,scaled_Laplacian

class Spatial(nn.Module):
    def __init__(self,close_size,k,N,model_d):
        super(Spatial,self).__init__()
        self.spatial = make_model(close_size,close_size,N,model_d,spatial=True)
        self.k = k  
    def forward(self,x_c,x_c_2,mode,A=None,index=None):
        '''initial data size
        x_c: bs*closeness*2*N
        x:   bs*2*N*closeness
        '''
        '''spatial input
        sx_c: bs*2*N*k*closeness
        tgt: bs*2*N*1*closeness
        '''
        '''spatial output
        sq_c: bs*N*1*closeness
        '''
        bs,closeness,_,N = x_c.shape#1*3*2*400
        #print('x_c.shape',x_c.shape)
        x = x_c.permute((0,2,3,1)).float()#1*2*400*3
        #print('x',x.shape)
        #calculate the similarity between other nodes
        if A is None:
            if(mode=='cos'):
                A = getA_cosin(x)
            elif(mode=='moran'):
                A = getA_Moran(x)
            elif(mode=='corr'):#maybe need absolute
                A = getA_corr(x)
            else:
                raise Exception('wrong adj mode')

        #print('A',A.shape)
        #selected top-k node
        sx_c = torch.zeros((bs,N,self.k,closeness),dtype=torch.float32)#1*400*20*3
        #print('sx_c',sx_c.shape)
        if index is None:
            index = torch.argsort(A,dim=-1,descending=True)[:,:,0:self.k] #bs,N,k
        # selected_c = []
        for i in range(bs):
            for j in range(N):
                sx_c[i,j] = x[i,0,index[i,j]]#1*400*20*3
        #sx_c = torch.cat(selected_c,dim=2).cuda()
        #print('sx_c',sx_c.shape)
        #sx_c:(bs,N,k,closeness)      
            
        if x_c_2 is None:
            tgt = x[:,0].unsqueeze(dim=-2).cuda()
        else:
            tgt=torch.mean(x_c_2[:,:,:,0],dim=1).transpose(1,2).unsqueeze(-2).cuda()#为啥要取平均？
        #print('tgt',tgt.shape)#[2, 1, 3, 1, 4000]

        sq_c = self.spatial(sx_c.cuda(), tgt).squeeze(dim=-2)#输入模型，一个是source sequence1*400*20*3，一个是initial sequence1*400*1*3
        #print('sq_c',sq_c.shape)
        #return sq_c.permute((0,3,1,2)) 
        #return F.sigmoid(sq_c).permute((0,3,1,2))
        return sq_c,tgt.permute((0,2,3,1)).squeeze(1)#输出sq_c1*400*3和1*3*400
 