import pandas as pd

from .dataset import Dataset
from .downloader import download_and_extract_zip


class SGEMMGPU(Dataset):
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00440/sgemm_product_dataset.zip"
    download_file_name = "sgemm_product_dataset.zip"
    extract_file_name = "sgemm_product.csv"

    def download_and_read_data(self):
        file_path = download_and_extract_zip(self.data_url, self.dataset_dir,
                                             self.download_file_name, self.extract_file_name)
        df = pd.read_csv(file_path)
        cols_y = [f"Run{i} (ms)" for i in range(1, 5)]
        self.y = df[cols_y].median(axis=1).to_numpy().ravel()
        df.drop(cols_y, axis=1, inplace=True)
        self.X = df.to_numpy()
        self.categorical_indicator = [False] * self.X.shape[1]
        self.attribute_names = list(df.columns)
