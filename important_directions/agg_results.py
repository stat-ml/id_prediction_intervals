import os
import glob

import numpy as np
import pandas as pd


def agg_results_dataset(path, subdir, start=1, stop=20):
    dfs = []
    for i in range(start, stop + 1):
        df = pd.read_csv(os.path.join(path, subdir, f"predictions_unscaled_{i}.csv"))
        df["run_id"] = i
        df["dataset"] = subdir
        dfs.append(df)
    return pd.concat(dfs)


def agg_results(path, start=1, stop=20):
    dataset_names = next(os.walk(path), (None, None, []))[1]
    #dataset_names = [os.path.splitext(f)[0] for f in filenames if f.endswith('yaml')]
    dfs = []
    for name in dataset_names:
        df_dataset = agg_results_dataset(path, name, start=start, stop=stop)
        dfs.append(df_dataset)
    df = pd.concat(dfs)
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    agg_results('./results')
