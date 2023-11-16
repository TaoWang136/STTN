import math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.linalg import eigs

def pr_subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def per_subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.zeros(np.ones(attn_shape)).astype('uint8')
    subsequent_mask[:,:,0] = 1
    return torch.from_numpy(subsequent_mask) == 0


def corr(x):
    (bs,number,N,c) = x.shape
    #print('x.shape':x.shape)
    x = x.transpose(1,2).contiguous().view((bs,N,c*number))#
    A = torch.zeros((bs,N,N),dtype=torch.float32,requires_grad=False)
    for i in range(bs):
        A[i] = torch.from_numpy(np.absolute(np.corrcoef(x[i].numpy())))
    for j in range(N): 
        A[:,j,j] = -1e9
    return F.softmax(A.reshape(bs,1,-1),dim=-1).reshape(bs,N,N)

class MinMaxNorm01(object):
    """scale data to range [0, 1]"""
    def __init__(self):
        pass

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()
        print('Min:{}, Max:{}'.format(self.min, self.max))

    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min)
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = x * (self.max - self.min) + self.min
        return x

