from typing import Any

from pl_bolts.datamodules.sklearn_datamodule import SklearnDataModule, SklearnDataset

from important_directions.datasets.debug_printing import get_line_info


def safe_1d_x(func):
    def inner(t):
        if len(t.shape) == 1:
            return func(t.reshape((1, -1))).reshape(-1)
        else:
            return func(t)

    return inner


class TransformedSklearnDataModule(SklearnDataModule):

    def __init__(self, X_transform_cls: Any = None, y_transform_cls: Any = None, *args, **kwargs):
        self.X_transform_cls = X_transform_cls
        self.y_transform_cls = y_transform_cls

        if self.X_transform_cls is not None:
            self.X_transform = X_transform_cls()
        else:
            self.X_transform = None

        if self.y_transform_cls is not None:
            self.y_transform = y_transform_cls()
        else:
            self.y_transform = None

        super().__init__(*args, **kwargs)

    def _init_datasets(self, X, y, x_val, y_val, x_test, y_test):
        print(f"{get_line_info()} {y.dtype=}")
        if self.X_transform is not None:
            self.X_transform.fit(X)
            xtr = safe_1d_x(self.X_transform.transform)
            #xtr = self.X_transform.transform
        else:
            xtr = None

        if self.y_transform is not None:
            self.y_transform.fit(y.reshape(-1, 1))
            ytr = lambda t: ((t - self.y_transform.mean_[0]) / self.y_transform.scale_[0]).astype(t.dtype)
            #ytr = self.y_transform.transform
        else:
            ytr = None

        self.train_dataset = SklearnDataset(X, y,
                                            X_transform=xtr,
                                            y_transform=ytr)
        self.val_dataset = SklearnDataset(x_val, y_val,
                                          X_transform=xtr,
                                          y_transform=ytr)
        self.test_dataset = SklearnDataset(x_test, y_test,
                                           X_transform=xtr,
                                           y_transform=ytr)

    def train_dataloader_fixed(self):
        is_shuffle = self.shuffle
        self.shuffle = False
        dl = self.train_dataloader()
        self.shuffle = is_shuffle

        return dl
