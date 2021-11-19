import numbers

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from .dataloader import BatchXYDataLoader


def max_k_n_ratio(k, n, ratio):
    return max(k, int(n * ratio))


def get_index_range(a, idx, start, stop):
    return a[idx[start:stop]]


def scale_inplace(t, mean, std):
    torch.sub(t, mean, out=t)
    torch.div(t, std, out=t)
    return t


def safe_numpy(t):
    """
    Safely transfer Torch tensor's *value* to numpy.
    """
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    x = t.numpy().copy()
    return x


class RegressionDataModule:
    def __init__(self, X, y, batch_size=128, device='cpu', seed=0, val_size=0.2, test_size=0.1,
                 torch_dtype=torch.float32, min_std=1e-6):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.val_size = val_size
        self.test_size = test_size
        self.torch_dtype = torch_dtype
        self.min_std = min_std

        # Splitting data
        self.random_state = None
        self.idx = None
        self.n_total = self.X.shape[0]
        self.n_model = None
        self.n_train = None
        self.n_val = None
        self.n_test = None

        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.X_train_mean = None
        self.X_train_std = None

        self.y_train_mean = None
        self.y_train_std = None

        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

        self.split_data()
        self.create_dataloaders()

    def split_data(self):
        if isinstance(self.test_size, numbers.Integral):
            self.n_test = self.test_size
        elif self.test_size > 0:
            self.n_test = max_k_n_ratio(1, self.n_total, self.test_size)
        else:
            self.n_test = 0

        # Number of instances left to prepare the models. Can take some for a validation set
        self.n_model = self.n_total - self.n_test

        if isinstance(self.val_size, numbers.Integral):
            self.n_val = self.val_size
        elif self.test_size > 0:
            self.n_val = max_k_n_ratio(1, self.n_model, self.val_size)
        else:
            self.n_val = 0

        self.n_train = self.n_model - self.n_val

        assert self.n_train > 0

        self.random_state = np.random.RandomState(self.seed)
        self.idx = np.arange(self.n_total)
        self.random_state.shuffle(self.idx)

        #print(f"{self.n_train=}, {self.n_model=}")

        limits = ((0, self.n_train), (self.n_train, self.n_model), (self.n_model, self.n_total))
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            (torch.as_tensor(get_index_range(a, self.idx, l[0], l[1]),
                             dtype=self.torch_dtype,
                             device=self.device) if l[1] - l[0] > 0 else None for a in (self.X, self.y) for l in limits)

        self.calculate_mean_std()

        for t in self.X_train, self.X_val, self.X_test:
            if t is not None:
                scale_inplace(t, self.X_train_mean, self.X_train_std)

        for t in self.y_train, self.y_val, self.y_test:
            if t is not None:
                scale_inplace(t, self.y_train_mean, self.y_train_std)

    def calculate_mean_std(self):
        (self.X_train_mean, self.X_train_std), (self.y_train_mean, self.y_train_std) = \
            map(lambda a: (a.mean(axis=0), torch.clip(a.std(axis=0), min=self.min_std, max=None)),
                (self.X_train, self.y_train))

    def create_dataloaders(self):
        self.train_loader, self.val_loader, self.test_loader = \
            map(lambda a, b, s: BatchXYDataLoader(a, b,
                                                  batch_size=self.batch_size,
                                                  shuffle=s) if a is not None and b is not None else None,
                (self.X_train, self.X_val, self.X_test),
                (self.y_train, self.y_val, self.y_test),
                (True, False, False))

    def get_y_test_numpy(self):
        return get_index_range(self.X, self.idx, self.n_model, self.n_total)

    def merge_validation(self):
        self.val_size = 0
        del self.X_train, self.y_train, \
            self.X_val, self.y_val, \
            self.X_train_std, self.X_train_mean, \
            self.y_train_std, self.y_train_mean

        if self.device != 'cpu':
            torch.cuda.empty_cache()

        self.split_data()
        self.create_dataloaders()


if __name__ == "__main__":
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    rdm = RegressionDataModule(X, y)
    tensors = list((rdm.X_train, rdm.X_val, rdm.X_test, rdm.y_train, rdm.y_val, rdm.y_test))
    #print(tensors)
    print(list(f"{a.shape}" for a in tensors))
    print(list(f"{a}" for a in (rdm.X_train_mean, rdm.X_train_std, rdm.y_train_mean, rdm.y_train_std)))
