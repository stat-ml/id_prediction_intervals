import argparse
import os

import numpy as np
import pandas as pd
import torch.multiprocessing as mp
import yaml
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

from important_directions.datasets.regression_datamodule import RegressionDataModule, safe_numpy
from important_directions.method_dropout import method_dropout
from important_directions.method_id import method_id
from important_directions.method_id_exact import method_id_exact
from important_directions.method_ens import method_ens

from important_directions.metrics import get_cov_prob


# from bnn_uncertainty.nnpi.nnpi import PredictionIntervals, PredictionIntervalsID


def run_experiment(config, args):
    seed = args.seed
    device = 'cuda:0' if args.gpu else 'cpu'
    # torch_dtype = torch.float32
    method_name = "imp_dirs"

    results_dir = os.path.join(os.getcwd(), "results", config["dataset_name"], args.method_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print(f"Loading dataset {config['dataset_name']}")
    try:
        from importlib import import_module

        datasets_module = import_module('important_directions.datasets')
        dataset_class = getattr(datasets_module, config["dataset_type"])
        dataset = dataset_class(**config)
    except AttributeError:
        print(f"Dataset type <{config['dataset_type']}> not found.")
        exit(1)

    dataset.load()

    data_module = RegressionDataModule(dataset.X, dataset.y, batch_size=config["batch_size"],
                                       seed=args.seed, device=device)

    if args.method_name == 'id':
        method = method_id
    elif args.method_name == 'dropout':
        method = method_dropout
    elif args.method_name == 'ide':
        method = method_id_exact
    elif args.method_name == 'ens':
        method = method_ens
    else:
        print(f"Method <{args.method_name}> not recognised!")
        exit(1)

    y_pred, y_l, y_u, y_true, additional = method(args, config, data_module, seed, device, results_dir)

    predictions = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "y_l": y_l, "y_u": y_u})
    predictions.to_csv(os.path.join(results_dir, f"predictions_{seed}.csv"), index=False)

    results = compute_metrics(args, config, data_module, y_pred, y_l, y_u, y_true, additional)

    print(results)
    results.to_csv(os.path.join(results_dir, f"results_{seed}.csv"), index=False)


def compute_metrics(args, config, rdm, y_pred, y_l, y_u, y_true, additional):
    widths = y_u - y_l
    r, p_val = pearsonr(widths, np.abs(y_true - y_pred))
    w_sd = widths.mean() / safe_numpy(rdm.y_train_std).item()
    #name = config.get("dataset_display_name", config["dataset_name"])
    name = config["dataset_name"]
    row = {"dataset": name, "split": args.seed, "method": args.method_name,
           "mse": mean_squared_error(y_true, y_pred),
           "r2": r2_score(y_true, y_pred),
           "p_cov": get_cov_prob(y_pred, y_l, y_u, y_true)[1],
           "w_sd": w_sd,
           "pearson_r": r,
           "pearson_r_p": p_val}
    row.update(additional)
    results = pd.DataFrame([row])
    return results


if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='Approximate prediction intervals for deep learning')
    parser.add_argument('config_path', type=str, help='Path to the YAML config file')

    parser.add_argument('-m', "--method", type=str, default='id', action='store', dest='method_name',
                        help='Name of the method: id - Important Directions, dropout - MC Dropout')

    parser.add_argument('-s', "--seed", type=int, default=1234, action='store', dest='seed',
                        help='Seed value for train-test split')
    parser.add_argument('-g', "--gpu", default=False, action='store_true', dest='gpu',
                        help='Seed value for train-test split')
    parser.add_argument('-dg', "--data_gpu", default=False, action='store_true', dest='is_data_gpu',
                        help='Whether to load full dataset to GPU')
    parser.add_argument('-r', "--resume", default=False, action='store_true', dest='is_resume',
                        help='Whether to continue from a saved NN model.')

    args = parser.parse_args()
    print(args)
    file_name = args.config_path

    if not os.path.exists(file_name):
        print(f"Experiment configuration file {file_name} not found!")
    else:
        with open(file_name) as fp:
            config = yaml.load(fp)
        print(config)
        run_experiment(config, args)
