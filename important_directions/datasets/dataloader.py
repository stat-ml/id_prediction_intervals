import numpy as np

import torch
from torch.utils.data import DataLoader


class BatchXYDataLoader:
    def __init__(self, X, y, batch_size=128, shuffle=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n = self.X.shape[0]

        self.len = self.n // self.batch_size
        if not self.n % self.batch_size == 0:
            self.len += 1

        self.indices = np.arange(self.n)

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return self.len

    def __iter__(self):
        for i in range(0, self.n, self.batch_size):
            j = min(i + self.batch_size, self.n)
            mask = self.indices[i:j]
            yield self.X[mask], self.y[mask]
        if self.shuffle:
            np.random.shuffle(self.indices)
