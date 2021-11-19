import logging
import os

import numpy as np
import pytorch_lightning as pl
import scipy
import torch
from pytorch_lightning import loggers as pl_loggers
from scipy.special import logsumexp
from tqdm.auto import tqdm

from .datasets.regression_datamodule import RegressionDataModule, safe_numpy
from .network import RegressionNet

# hyperparameters to search
dropout_rates = [0.005, 0.01, 0.05, 0.1]
# tau_values = [0.1, 0.15, 0.2]
# tau_values = [1, 10, 100]
# tau_values = [4, 64, 4096]

implied_sigma = np.array([0.5, 0.25, 0.1])
tau_values = 1. / implied_sigma ** 2
max_epochs = 400

batch_size = 128
lengthscale = 1e-2


def method_dropout(args, config, data_module: RegressionDataModule, seed, device, results_dir):
    method_name = "dropout"
    # ckpt_dir = os.path.join(results_dir, f"{method_name}")
    ckpt_dir = results_dir
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    best_network = None
    best_ll = -float('inf')
    best_tau = 0
    best_dropout = 0
    # While doing grid search, disable excessive logging
    ptl_logger = logging.getLogger("pytorch_lightning")
    old_log_level = ptl_logger.level
    ptl_logger.setLevel(logging.ERROR)

    N = data_module.X_train.shape[0]

    def create_and_fit_model(p, tau, dataloader, checkpoint_path, logger=False, weights_summary=None):
        weight_decay = lengthscale ** 2 * (1 - p) / (2. * N * tau)

        net = RegressionNet(sizes=config["layer_sizes"],
                            num_inputs=data_module.X.shape[1],
                            nonlin='LeakyReLU',
                            optimizer_name='Adam',
                            lr=config["lr"],
                            weight_decay=weight_decay)

        grid_trainer = pl.Trainer(logger=logger,
                                  max_epochs=max_epochs,
                                  gpus=1 if not device == "cpu" else 0,
                                  log_every_n_steps=1,
                                  weights_summary=weights_summary)

        grid_trainer.fit(net, train_dataloaders=dataloader)

        grid_trainer.save_checkpoint(checkpoint_path)

        return net

    for i, dropout_rate in enumerate(dropout_rates):
        for j, tau_value in enumerate(tau_values):
            print(f"Grid search step: Tau: {tau_value:.2f} + Dropout rate: {dropout_rate:.2f}")
            ckpt_path = os.path.join(ckpt_dir, f"model_{seed}_{i}_{j}.ckpt")
            model = create_and_fit_model(dropout_rate, tau_value, data_module.train_loader, ckpt_path)

            y_pred, y_l, y_u, y_true, ll, mse, mse_loss = predict_mc_to_numpy(model, data_module.val_loader,
                                                                              tau=tau_value,
                                                                              T=10000,
                                                                              device=device)

            print(f"{mse=}, {ll=}")
            if ll > best_ll:
                best_ll = ll
                best_tau = tau_value
                best_dropout = dropout_rate
                print(f'New best log_likelihood: {best_ll}, best tau changed to: {best_tau}, '
                      f'best dropout rate changed to: {best_dropout}')
    # Restore logging
    ptl_logger.setLevel(old_log_level)

    # Refit the best model
    data_module.merge_validation()

    tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/',
                                             name=f'{config["dataset_name"]}/split_{seed}/{method_name}')
    ckpt_path = os.path.join(ckpt_dir, f"model_{seed}.ckpt")
    print(f"Best model found: {best_dropout=} + {best_tau=}")
    best_model = create_and_fit_model(best_dropout, best_tau, data_module.train_loader, ckpt_path,
                                      logger=tb_logger, weights_summary="top")

    ym = safe_numpy(data_module.y_train_mean).item()
    ys = safe_numpy(data_module.y_train_std).item()

    y_pred, y_l, y_u, y_true, ll, _, _ = predict_mc_to_numpy(best_model, data_module.test_loader, tau=best_tau,
                                                             y_mean=ym, y_std=ys, T=10000, device=device)

    return y_pred, y_l, y_u, y_true, {"dropout_rate": best_dropout, "tau": best_tau}


def predict_mc_to_numpy(model, dataloader, tau, y_mean=0., y_std=1., T=10000, device='cpu'):
    prediction_list = []
    y_test_batches = []

    model.to(device)
    model.train()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Dropout predict")

    with torch.no_grad():
        for i, batch in pbar:
            x, y = batch
            n = x.shape[0]
            y_test_batches.append(safe_numpy(y))
            y_preds = torch.zeros((n, T))
            for j in range(T):
                y_preds[:, j] = model(x)
            prediction_list.append(safe_numpy(y_preds))
    y_pred_all = np.concatenate(prediction_list)
    # print(f"{y_pred_all.shape=}")
    y_test = np.concatenate(y_test_batches).ravel()
    y_pred = y_pred_all.mean(axis=1).ravel()

    total_unc = np.sqrt(y_pred_all.var(axis=1) + 1. / tau).ravel()
    # ll_proper = -scipy.stats.norm(loc=y_pred, scale=total_unc).logpdf(y_test.ravel()).mean()

    ll = (logsumexp(-0.5 * tau * (y_test[:, None] - y_pred_all) ** 2., axis=1) - np.log(T)
          - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(tau))
    test_ll = np.mean(ll)

    ll_proper = test_ll

    y_l = scipy.stats.norm.ppf(0.025, loc=y_pred, scale=total_unc).ravel()
    y_u = scipy.stats.norm.ppf(0.975, loc=y_pred, scale=total_unc).ravel()

    # MSE in the same scale as training
    mse_loss = ((y_pred - y_test) ** 2).mean()

    # Return to original scale
    y_pred, y_l, y_u, y_test = map(lambda t: t * y_std + y_mean, (y_pred, y_l, y_u, y_test))

    # MSE in original dataset's units
    mse = ((y_pred - y_test) ** 2).mean()

    # ncov, pcov = get_cov_prob(y_pred, y_l, y_u, y_test.ravel())
    # print("Mean total unc = {}, ll_proper = {}, pcov = {}".format(total_unc.mean(), ll_proper, pcov))

    # df = pd.DataFrame({"y_pred": y_pred, "y_l": y_l, "y_u": y_u, "y_true": y_test.ravel(), "y_std": total_unc})

    model.to('cpu')

    return y_pred, y_l, y_u, y_test, ll_proper, mse, mse_loss
