import os
import pandas as pd

from .dataset import Dataset
from .downloader import download_file_to_dir


class ProteinStructure(Dataset):
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
    download_file_name = "CASP.csv"

    def download_and_read_data(self):
        download_path = download_file_to_dir(self.data_url, self.dataset_dir, self.download_file_name)

        df = pd.read_csv(download_path)
        m = df.to_numpy()
        self.X = m[:, 1:]
        self.y = m[:, 0]
        self.categorical_indicator = [False] * self.X.shape[1]
        self.attribute_names = list(df.columns[1:])
