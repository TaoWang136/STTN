import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

def load_adj(dataset_name):
    # dataset_path = './data'
    # dataset_path = os.path.join(dataset_path, dataset_name)
    # adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    # adj = adj.tocsc()
    # print('adj',adj)
    # n_vertex = 228

    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    adj = np.load(os.path.join(dataset_path, 'adj_100.npy'))
    adj = sp.csr_matrix(adj)
    print('adj',adj)
    n_vertex = adj.shape[0]  
    
    
    return adj, n_vertex

def load_data(dataset_name, len_train, len_val):
    dataset_path = './data'
    dataset_path = os.path.join(dataset_path, dataset_name)
    #vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))
    vel = pd.read_csv(os.path.join(dataset_path, 'data_risk_10_100.csv'))
    print(vel)
    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    print('test',test.shape)
    return train, val, test

def data_transform(data, n_his, n_pred, device):
    # produce data slices for x_data and y_data

    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred
    
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])
    
    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]
        # print('xxxxxxxxx',x.shape)
        # print('yyyyyyyyyyyy',y.shape)
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)