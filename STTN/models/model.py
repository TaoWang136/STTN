import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from .spatiotemporal import proximal,period,Spatial
from .utils import corr

class Fusion(nn.Module):
    def __init__(self,dim_in):
        super(Fusion,self).__init__()
        self.weight2 = nn.Linear(dim_in*2,dim_in)

    def forward(self,x1,x2=None):
        if(x2 is not None):
            out = self.weight2(torch.cat([x1,x2],dim=-1))
        else:
            out = x1
        return out

class STTN(nn.Module):
    def __init__(self,len_proximal, external_size, N, k, spatial, s_model_d,pr_model_d,p_model_d,dim_hid=64):
        super(STTN,self).__init__()

        self.spatial = Spatial(len_proximal,k,N,s_model_d)#输出400*3
        self.pr_temporal = proximal(k,N,pr_model_d)
        self.p_temporal = period(len_proximal,N,p_model_d)
        self.temporal_fusion = Fusion(len_proximal)
        self.spatial_f = Spatial(len_proximal,k,N,s_model_d)
        self.fusion = Fusion(len_proximal)
        self.k = k
    def forward(self,x_pr,mode,pr,s,FS,pr_tgt,s_tgt,number,x_p):
        bs = len(x_pr)
        N = x_pr.shape[-1]
        len_proximal = x_pr.shape[1]
        x_spatial = None
        sq_t,sq_p,sq_pr = None,None,None
        adj = corr(x_pr.permute((0,2,3,1)))
        index = torch.argsort(adj,dim=-1,descending=True)[:,:,0:self.k]
        if(s):
            x_spatial,_ = self.spatial(x_pr,x_p,s_tgt,mode,number,adj,index)
        if(pr):
            sq_pr = F.sigmoid(self.pr_temporal(x_pr,x_p,pr_tgt,mode,number,adj,index))
        sq_p = self.p_temporal(x_pr, x_p,number)
        x_temporal = self.temporal_fusion(sq_p,sq_pr)
        if(FS):
            x_temporal,_ = self.spatial_f(x_pr,x_temporal.transpose(1,2).unsqueeze(-2).unsqueeze(1),'p',mode,number,adj,index)
        pred = self.fusion(x_temporal,x_spatial)
        return pred.transpose(1,2)



