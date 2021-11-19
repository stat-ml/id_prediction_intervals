import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping

from important_directions.datasets.regression_datamodule import RegressionDataModule, safe_numpy
from important_directions.imp_dirs import ImportantDirectionsPytorch
from important_directions.network import RegressionNet, predict_to_numpy


def method_id(args, config, data_module: RegressionDataModule, seed, device, results_dir):
    method_name = 'important_directions'
    N = data_module.X_train.shape[0]
    lmb = config["weight_decay"] * N / 2.0
    print(f"{N=}, {data_module.X.shape=}, {lmb=}, {config['weight_decay']=}")
    # train model
    model = RegressionNet(num_inputs=data_module.X_train.shape[1],
                          sizes=config["layer_sizes"],
                          nonlin='LeakyReLU',
                          optimizer_name="Adam",
                          lr=config["lr"],
                          weight_decay=config["weight_decay"])
    # print_model_weights(model)
    ckpt_path = os.path.join(results_dir, f"model_{seed}.ckpt")
    if args.is_resume and os.path.exists(ckpt_path):
        print(f"Using trained model from [{ckpt_path}]...")
        model = RegressionNet.load_from_checkpoint(ckpt_path)
    else:
        print(f"No model at [{ckpt_path}], training...")
        early_stopping = EarlyStopping('val_mse_epoch', patience=100)
        trainer = pl.Trainer(logger=False,
                             max_epochs=config.get("max_epochs", 10000),
                             gpus=1 if not device == "cpu" else 0,
                             log_every_n_steps=1,
                             callbacks=[early_stopping])

        trainer.fit(model, train_dataloaders=data_module.train_loader, val_dataloaders=data_module.val_loader)
        #print(f"{dir(trainer)=}, \n {dir(early_stopping)=}, {early_stopping.stopped_epoch=}")
        #exit(1)
        print(f"{early_stopping.stopped_epoch=}")

        # Train best network on full train set
        best_num_epochs = max(early_stopping.stopped_epoch, config.get("min_epochs", 200))
        data_module.merge_validation()
        tb_logger = pl_loggers.TensorBoardLogger(save_dir='logs/',
                                                 name=f'{config["dataset_name"]}/split_{seed}/{method_name}', )
        trainer = pl.Trainer(logger=tb_logger,
                             max_epochs=best_num_epochs,
                             gpus=1 if not device == "cpu" else 0,
                             log_every_n_steps=1)

        trainer.fit(model, data_module.train_loader)
        trainer.save_checkpoint(ckpt_path)

    imp_dirs = ImportantDirectionsPytorch(model, rank=config["max_rows"], alpha_final=lmb, device=device)
    imp_dirs.fit(data_module.X_train, data_module.y_train)

    imp_dirs_path = os.path.join(results_dir, f"u_d_vt_{seed}.npz")
    imp_dirs.save_to_file(imp_dirs_path)

    y_pred_, y_l, y_u = imp_dirs.predict_to_numpy(data_module.X_test, data_module.y_test)
    y_true = safe_numpy(data_module.y_test)
    y_pred = predict_to_numpy(model, data_module.test_loader, device=device)

    #print(f"{np.max(np.abs(y_pred_ - y_pred))=}")
    ym = safe_numpy(data_module.y_train_mean).item()
    ys = safe_numpy(data_module.y_train_std).item()

    # return to original scale
    y_pred, y_l, y_u, y_true = map(lambda y: y * ys + ym, (y_pred, y_l, y_u, y_true))

    return y_pred, y_l, y_u, y_true, {"p_star": imp_dirs.p_star, "rank": imp_dirs.rank, "M": imp_dirs.m,
                                      "s_hat": imp_dirs.s_hat, "tq": imp_dirs.tq}
