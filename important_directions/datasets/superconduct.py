import pandas as pd

from .dataset import Dataset
from .downloader import download_and_extract_zip


class Superconduct(Dataset):
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip"
    download_file_name = "superconduct.zip"
    extract_file_name = "train.csv"

    def download_and_read_data(self):
        file_path = download_and_extract_zip(self.data_url, self.dataset_dir,
                                             self.download_file_name, self.extract_file_name)
        df = pd.read_csv(file_path)
        m = df.to_numpy()
        self.X = m[:, :-1]
        self.y = m[:, -1]
        self.categorical_indicator = [False] * self.X.shape[1]
        self.attribute_names = list(df.columns[:-1])
