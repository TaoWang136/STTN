# -*- coding: utf-8 -*-
"""
/*******************************************
**  license
********************************************/
"""
import pandas as pd
import numpy as np
import numpy as np
from pandas import to_datetime
from sttn_prediction.models.utils import MinMaxNorm01
# from sttn_prediction.dataloader.data_precess import data_precess
import torch
import pandas as pd

def load_data(proximal_size, period_size, len_test,SSM):
    kpe=pd.read_csv('/home/deepcoder/桌面/wangtao/德国数据时空数据预测/%s_risk_16s.csv'%SSM)#修改这里
    #kpe.drop(['frame'],axis=1,inplace=True)
    all_intera_1=kpe.values
    arr1 = all_intera_1[:, np.newaxis,:]#升维变成1488*1*245    
    data=arr1
    index=pd.date_range(start='2023-11-01 00:00:00',periods=data.shape[0],freq='H')

    data_all = [data]
    index_all = [index]

    mmn = MinMaxNorm01()
    data_train = data[:-len_test]
    mmn.fit(data_train)
    data_all_mmn = []
    for data in data_all:
        data_all_mmn.append(mmn.transform(data))

    xpr, xp,  = [], []
    y = []
    timestamps_y = []
    for data, index in zip(data_all_mmn, index_all):
        #print(data.shape,index.shape) #(1488,2,400) (1488,)
        st = data_precess(data, index, 260)
        _xpr, _xp, _y, _timestamps_y = st.create_dataset(
            len_proximal=proximal_size, len_period=period_size,PeriodInterval=1)
        # print('_xc:',_xc.shape,'_xp:',_xp.shape,'_xt:',_xt.shape)
        xpr.append(_xpr)
        xp.append(_xp)
        y.append(_y)
        timestamps_y += _timestamps_y
    
    
    xpr = np.vstack(xpr)
    xp = np.vstack(xp)
    y = np.vstack(y)
    
    xpr_train, xp_train, y_train = xpr[:-len_test], xp[:-len_test], y[:-len_test]
    xpr_test, xp_test, y_test = xpr[-len_test:], xp[-len_test:], y[-len_test:]
    timestamps_train, timestamps_test = timestamps_y[:-len_test], timestamps_y[-len_test:]

    x_train = []
    x_test = []

    for l, x_ in zip([proximal_size, period_size], [xpr_train, xp_train]):
        if l > 0:
            x_train.append(x_)

    for l, x_ in zip([proximal_size, period_size], [xpr_test, xp_test]):
        if l > 0:
            x_test.append(x_)
    return x_train, y_train, x_test, y_test, mmn


class data_precess(object):
    def __init__(self, data, timestamps, T, CheckComplete=True):
        super(data_precess, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps #list[(index,timestamp)]
        self.T = T
        self.pd_timestamps = timestamps
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def get_matrix(self, timestamp, len_proximal, proximal):
        index = self.get_index[timestamp]
        if(proximal):
            return self.data[index]
        else:
            return self.data[np.arange(index,index+len_proximal)]


    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_proximal=3, len_period=2, PeriodInterval=1):
        offset_frame = pd.DateOffset(hours=1)
        XPr = []
        XP = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_proximal+1),
                   [PeriodInterval * self.T * j for j in range(1, len_period+1)]
                   ]
        i = max( self.T * PeriodInterval * len_period, len_proximal)
        #print('i:',i)
        while i < len(self.pd_timestamps)-len_proximal:
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
            x_pr = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame,len_proximal,proximal=True) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame,len_proximal,proximal =False) for j in depends[1]]
         
            y = self.get_matrix(self.pd_timestamps[i],len_proximal,proximal=False)
            #print(x_c[0].shape,x_p[0].shape) #(2,400) (6,2,400)
            if len_proximal > 0:
                XPr.append(np.stack(x_pr))
            if len_period > 0:
                XP.append(np.stack(x_p))

            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
       # print(timestamps_Y)
        XPr = np.asarray(XPr)
        XP = np.asarray(XP)
        Y = np.asarray(Y)
        print("XPr shape: ", XPr.shape, "XP shape: ", XP.shape, "Y shape:", Y.shape)
        return XPr, XP, Y, timestamps_Y
