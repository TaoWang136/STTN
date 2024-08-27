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
from sttn_git.models.utils import MinMaxNorm01
import os
import torch
import pandas as pd


class risk_data:
    def __init__(self,proximal_size, period_size, len_test, SSM):
        self.SSM=SSM
        self.proximal_size=proximal_size
        self.period_size=period_size
        self.len_test=len_test
        self.SSM=SSM
        self.mmn=MinMaxNorm01()
    def load_and_process(self):
        # Construct file path using os.path.join for better compatibility
        file_path = os.path.join('./dataloader', f'{self.SSM}_risk_16s.csv')
        kpe = pd.read_csv(file_path)
        # Add a new axis to match the expected shape (1488, 1, 245)
        data = kpe.values[:, np.newaxis, :]
        # Create a datetime index starting from a specific point
        index = pd.date_range(start='2023-11-01 00:00:00', periods=data.shape[0], freq='H')
        # Initialize the MinMax Normalizer and normalize the training data

        data_train = data[:-self.len_test]
        self.mmn.fit(data_train)
        data_normalized = self.mmn.transform(data)
        # Setup data processing for creating the dataset
        self.data_processor = data_process(data_normalized, index, 260)
        xpr, xp, y, timestamps_y = self.data_processor.create_dataset(
            len_proximal=self.proximal_size, len_period=self.period_size, PeriodInterval=1)
        # Split data into training and test sets
        xpr_train, xp_train, y_train = xpr[:-self.len_test], xp[:-self.len_test], y[:-self.len_test]
        xpr_test, xp_test, y_test = xpr[-self.len_test:], xp[-self.len_test:], y[-self.len_test:]
        # Organize the data based on provided sizes for proximal and period parameters
        x_train = [x for x, l in zip([xpr_train, xp_train], [self.proximal_size, self.period_size]) if l > 0]
        x_test = [x for x, l in zip([xpr_test, xp_test], [self.proximal_size, self.period_size]) if l > 0]
        return x_train, y_train, x_test, y_test, self.mmn

class data_process:
    def __init__(self, data, timestamps, T):
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = pd.to_datetime(timestamps)
        self.T = T
        self.index_map = {ts: i for i, ts in enumerate(self.timestamps)}

    def get_matrix(self, timestamp, len_proximal, is_proximal):
        index = self.index_map[timestamp]
        return self.data[index] if is_proximal else self.data[index:index + len_proximal]

    def create_dataset(self, len_proximal=3, len_period=2, PeriodInterval=1):
        offset = pd.DateOffset(hours=1)
        XPr, XP, Y, timestamps_Y = [], [], [], []
        for i in range(max(self.T * PeriodInterval * len_period, len_proximal), len(self.timestamps) - len_proximal):
            ts = self.timestamps[i]
            depends = [
                [(ts - j * offset) for j in range(1, len_proximal + 1)],
                [(ts - PeriodInterval * self.T * j * offset) for j in range(1, len_period + 1)]
            ]
            if all(d in self.index_map for dep in depends for d in dep):
                x_pr = [self.get_matrix(d, len_proximal, True) for d in depends[0]]
                x_p = [self.get_matrix(d, len_proximal, False) for d in depends[1]]
                y = self.get_matrix(ts, len_proximal, False)
                XPr.append(np.stack(x_pr))
                XP.append(np.stack(x_p))
                Y.append(y)
                timestamps_Y.append(ts)

        return np.array(XPr), np.array(XP), np.array(Y), timestamps_Y
