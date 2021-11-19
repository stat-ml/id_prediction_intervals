import os

import numpy as np
import openml


class Dataset:
    def __init__(self, dataset_name, save_dir, openml_id=None, *args, **kwargs):
        self.dataset_name = dataset_name

        self.dataset_dir = os.path.join(os.getcwd(), save_dir, dataset_name)
        self.x_path = os.path.join(self.dataset_dir, "X.npy")
        self.y_path = os.path.join(self.dataset_dir, "y.npy")
        self.cat_path = os.path.join(self.dataset_dir, "categorical_indicator.txt")
        self.attr_names_path = os.path.join(self.dataset_dir, "attribute_names.txt")

        self.X = None
        self.y = None
        self.categorical_indicator = None
        self.attribute_names = None

    def load(self):
        if all(os.path.exists(path) for path in (self.x_path, self.y_path, self.cat_path, self.attr_names_path)):
            self.read_data()
        else:
            print(f"Datasets {self.dataset_name} is missing, downloading...")
            if not os.path.isdir(self.dataset_dir):
                os.makedirs(self.dataset_dir)
            self.download_and_read_data()
            print(f"Finished downloading {self.dataset_name}, saving...")
            self.save_data()
            print(f"Finished saving {self.dataset_name}, done.")

    def download_and_read_data(self):
        raise NotImplementedError

    def save_data(self):
        np.save(self.x_path, self.X)
        np.save(self.y_path, self.y)
        np.savetxt(self.cat_path, self.categorical_indicator, fmt='%s')
        np.savetxt(self.attr_names_path, self.attribute_names, fmt='%s')

    def read_data(self):
        self.X = np.load(self.x_path)
        self.y = np.load(self.y_path)
        self.categorical_indicator = np.genfromtxt(self.cat_path, dtype=bool)
        self.attribute_names = np.genfromtxt(self.attr_names_path, dtype=str, delimiter='\n', usecols=[0])

        #print(f"{self.attribute_names=}")
        #print(f"{np.isnan(self.X).sum()=}")


class OpenMLDataset(Dataset):
    def __init__(self, dataset_name, save_dir, openml_id=None, *args, **kwargs):
        super().__init__(dataset_name, save_dir)

        self.openml_id = openml_id

    def download_and_read_data(self):
        dataset = openml.datasets.get_dataset(self.openml_id)
        self.X, self.y, self.categorical_indicator, self.attribute_names = dataset.get_data(
            dataset_format="array", target=dataset.default_target_attribute
        )


if __name__ == "__main__":
    import yaml

    config = yaml.load(open("./configs/concrete.yaml", "r"))
    print(config)
    ds = OpenMLDataset(**config)

    #ds.load()
    ds.download_and_read_data()
    #print(ds.X, ds.y, ds.categorical_indicator, ds.attribute_names)
    print(f"{ds.X.shape=}", f"{ds.y.shape=}", ds.categorical_indicator, ds.attribute_names)
