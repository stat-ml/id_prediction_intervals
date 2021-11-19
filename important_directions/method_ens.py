import logging
import os

import numpy as np
import scipy
import scipy.stats

import torch.cuda

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

from important_directions.datasets.regression_datamodule import RegressionDataModule, safe_numpy
from important_directions.network import RegressionNet, predict_to_numpy


def safe_del_model(model, device):
    del model
    if device != 'cpu':
        torch.cuda.empty_cache()


def method_ens(args, config, data_module: RegressionDataModule, seed, device, results_dir):
    method_name = 'ensemble'

    # train ensemble

    # determine optimal number of epochs
    net = RegressionNet(sizes=config["layer_sizes"],
                        num_inputs=data_module.X.shape[1],
                        nonlin='LeakyReLU',
                        optimizer_name='Adam',
                        lr=config["lr"],
                        weight_decay=config["weight_decay"])

    early_stopping = EarlyStopping('val_mse_epoch', patience=100)
    val_trainer = pl.Trainer(logger=False,
                             max_epochs=config.get("max_epochs", 10000),
                             gpus=1 if not device == "cpu" else 0,
                             log_every_n_steps=1,
                             callbacks=[early_stopping])
    val_trainer.fit(net, train_dataloaders=data_module.train_loader, val_dataloaders=data_module.val_loader)

    val_trainer.save_checkpoint(os.path.join(results_dir, f"model_{seed}_val.ckpt"))
    best_num_epochs = max(early_stopping.stopped_epoch, config.get("min_epochs", 200))
    y_preds_val = predict_to_numpy(net, data_module.val_loader, device=device)
    y_val = safe_numpy(data_module.y_val)
    y_xi2 = ((y_val - y_preds_val) ** 2).mean()

    safe_del_model(net, device)

    # create full train set
    data_module.merge_validation()

    M = 5

    y_preds = np.zeros((M, len(data_module.y_test)))

    # While doing grid search, disable excessive logging
    ptl_logger = logging.getLogger("pytorch_lightning")
    old_log_level = ptl_logger.level
    ptl_logger.setLevel(logging.ERROR)

    for i in range(M):
        # train individual model
        ckpt_path = os.path.join(results_dir, f"model_{seed}_{i}.ckpt")

        model = RegressionNet(num_inputs=data_module.X_train.shape[1],
                              sizes=config["layer_sizes"],
                              nonlin='LeakyReLU',
                              optimizer_name="Adam",
                              lr=config["lr"],
                              weight_decay=config["weight_decay"])
        # print_model_weights(model)
        if args.is_resume and os.path.exists(ckpt_path):
            print(f"Using trained model from [{ckpt_path}]...")
            model = RegressionNet.load_from_checkpoint(ckpt_path)
        else:
            print(f"No model at [{ckpt_path}], training...")
            tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/',
                                                     name=f'{config["dataset_name"]}/split_{seed}/{method_name}', )
            trainer = pl.Trainer(logger=tb_logger,
                                 max_epochs=best_num_epochs,
                                 gpus=1 if not device == "cpu" else 0,
                                 log_every_n_steps=1,
                                 weights_summary=None)

            trainer.fit(model, data_module.train_loader)
            trainer.save_checkpoint(ckpt_path)

        y_preds[i] = predict_to_numpy(model, data_module.test_loader, device=device)
        safe_del_model(model, device)

    # Restore logging
    ptl_logger.setLevel(old_log_level)

    y_true = safe_numpy(data_module.y_test)

    y_pred = y_preds.mean(axis=0)
    y_std = np.sqrt(y_preds.var(axis=0) + y_xi2)

    y_l, y_u = scipy.stats.norm.interval(alpha=0.95, loc=y_pred, scale=y_std)

    ym = safe_numpy(data_module.y_train_mean).item()
    ys = safe_numpy(data_module.y_train_std).item()

    #print(list(x.shape for x in (y_pred, y_l, y_u, y_true)))

    # return to original scale
    y_pred, y_l, y_u, y_true = map(lambda y: y * ys + ym, (y_pred, y_l, y_u, y_true))

    return y_pred, y_l, y_u, y_true, {"y_xi2": y_xi2, "y_xi": np.sqrt(y_xi2)}
