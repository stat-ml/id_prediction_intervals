import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from tqdm.auto import tqdm

import pytorch_lightning as pl
from torchmetrics.regression import ExplainedVariance, MeanSquaredError

from .datasets.regression_datamodule import safe_numpy


class RegressionNet(pl.LightningModule):
    def __init__(self, num_inputs, sizes=(10,), nonlin='LeakyReLU',
                 optimizer_name="SGD", lr=1e-3, weight_decay=0, dropout_rate=None):
        super().__init__()

        self.save_hyperparameters()

        self.num_inputs = self.hparams.num_inputs
        self.sizes = self.hparams.sizes
        self.optimizer_name = self.hparams.optimizer_name
        self.lr = self.hparams.lr
        self.weight_decay = self.hparams.weight_decay
        self.nonlin_class = getattr(nn, self.hparams.nonlin)
        self.dropout_rate = self.hparams.dropout_rate

        layers = [nn.Linear(self.num_inputs, self.sizes[0]), self.nonlin_class()]
        if self.dropout_rate is not None:
            layers.append(nn.Dropout(p=self.dropout_rate))
        for i in range(len(self.sizes) - 1):
            layers += [nn.Linear(self.sizes[i], self.sizes[i + 1]), self.nonlin_class()]
            if self.dropout_rate is not None:
                layers.append(nn.Dropout(p=self.dropout_rate))
        layers.append(nn.Linear(self.sizes[-1], 1))

        self.net = nn.Sequential(*layers)

        # Metrics
        self.train_ev = ExplainedVariance()
        self.train_mse = MeanSquaredError()

        self.val_ev = ExplainedVariance()
        self.val_mse = MeanSquaredError()

    def forward(self, x):
        return self.net(x).reshape(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.net(x).reshape(-1)

        mse = F.mse_loss(y_hat, y)
        # reg = 0
        # if self.lmb_wd > 0:
        #    for param in self.net.parameters():
        #        reg += (param ** 2).sum()
        # loss = mse + self.lmb_wd * reg
        loss = mse

        self.log('train_loss', loss)

        self.train_ev.update(y_hat, y)
        self.log('train_ev', self.train_ev, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.train_mse.update(y_hat, y)
        self.log('train_mse', self.train_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.net(x).reshape(-1)

        self.val_ev.update(y_hat, y)
        self.log('val_ev', self.val_ev, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.val_mse.update(y_hat, y)
        self.log('val_mse', self.val_mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_ev_epoch', self.train_ev.compute())
        self.log('train_mse_epoch', self.train_mse.compute())

    def on_validation_epoch_end(self):
        # log epoch metric
        self.log('val_ev_epoch', self.val_ev.compute())
        self.log('val_mse_epoch', self.val_mse.compute())

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(self.parameters(),
                                                              lr=self.lr,
                                                              weight_decay=self.weight_decay)
        return optimizer


def predict_to_numpy(model, dataloader, transform=None, device='cpu'):
    prediction_list = []

    model.to(device)

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y = batch
            output = model(x)
            prediction_list.append(safe_numpy(output))
    y_pred = np.concatenate(prediction_list).ravel()

    if transform is not None:
        y_pred = transform(y_pred)

    model.to('cpu')

    return y_pred


def create_and_train_model():
    pass
