import os

import numpy as np
import scipy
import scipy.stats

from tqdm.auto import tqdm

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping

from important_directions.datasets.regression_datamodule import RegressionDataModule, safe_numpy
from important_directions.imp_dirs import ImportantDirectionsPytorch, numel, flatten_grad
from important_directions.network import RegressionNet, predict_to_numpy


def method_id_exact(args, config, data_module: RegressionDataModule, seed, device, results_dir):
    #method_name = 'important_directions'
    N = data_module.X_train.shape[0]
    lmb = config["weight_decay"] * N / 2.0
    print(f"{N=}, {data_module.X.shape=}, {lmb=}, {config['weight_decay']=}")
    # train model
    ckpt_path = os.path.join(os.getcwd(), "results", config["dataset_name"], "id", f"model_{seed}.ckpt")
    model = RegressionNet.load_from_checkpoint(ckpt_path)
    M = numel(model)

    u_d_vt_path = os.path.join(os.getcwd(), "results", config["dataset_name"], "ide", f"u_d_vt_{seed}.npz")

    if os.path.exists(u_d_vt_path):
        data = np.load(u_d_vt_path)
        U = data["U"]
        d = data["d"]
        Vt = data["Vt"]

        y_pred_train = torch.from_numpy(
            predict_to_numpy(model, data_module.train_loader, device=device).astype(np.float32)
        ).to(device)
    else:
        JtJ_tensor = torch.zeros((M, M), dtype=torch.float32, device=device)
        gTg = torch.zeros_like(JtJ_tensor)
        y_pred_train = torch.zeros(N, dtype=torch.float32, device=device)
        model.to(device)
        for i in tqdm(range(N)):
            yi = model(data_module.X_train[i])
            y_pred_train[i] = yi.detach()
            g = flatten_grad(torch.autograd.grad(yi, model.parameters()))
            torch.outer(g, g, out=gTg)
            JtJ_tensor += gTg

        JtJ = safe_numpy(JtJ_tensor)
        del JtJ_tensor, gTg
        if device != 'cpu':
            torch.cuda.empty_cache()

        U, d, Vt = np.linalg.svd(JtJ, full_matrices=False)
        np.savez(u_d_vt_path, U=U, d=d, Vt=Vt)

    DV = d / (d + lmb) ** 2
    DH = d / (d + lmb)
    p_star_diag = 2 * DH - DH ** 2
    p_star = p_star_diag.sum()
    print(f"{lmb=}, {p_star=}, {M=}, {N=}")

    Vt_tensor = torch.from_numpy(Vt.astype(np.float32)).to(device=device)
    DV_tensor = torch.from_numpy(DV.astype(np.float32)).to(device=device)

    residuals2 = safe_numpy((y_pred_train - data_module.y_train) ** 2)

    s_hat = np.sqrt(residuals2.sum() / (N - p_star))
    tq = scipy.stats.t.ppf(0.975, N - p_star)

    #y_pred_, y_l, y_u = imp_dirs.predict_to_numpy(data_module.X_test, data_module.y_test)

    M_test = data_module.X_test.shape[0]
    y_pred_test = torch.zeros(M_test, dtype=torch.float32, device=device)
    width = torch.zeros_like(y_pred_test)

    model.to(device)
    for i in tqdm(range(M_test), disable=False):
        yi_pred = model(data_module.X_test[i])
        y_pred_test[i] = yi_pred.detach()
        gi = flatten_grad(torch.autograd.grad(yi_pred, model.parameters()))

        GV = torch.mv(Vt_tensor, gi)
        prod = (DV_tensor * ((GV) ** 2)).sum()
        sqrt_term = torch.sqrt(1 + prod)
        width[i] = tq * s_hat * sqrt_term

    y_l = safe_numpy(y_pred_test - width)
    y_u = safe_numpy(y_pred_test + width)
    y_true = safe_numpy(data_module.y_test)
    y_pred = safe_numpy(y_pred_test)

    ym = safe_numpy(data_module.y_train_mean).item()
    ys = safe_numpy(data_module.y_train_std).item()

    # return to original scale
    y_pred, y_l, y_u, y_true = map(lambda y: y * ys + ym, (y_pred, y_l, y_u, y_true))

    return y_pred, y_l, y_u, y_true, {"p_star": p_star, "rank": M, "M": M,
                                      "s_hat": s_hat, "tq": tq}
