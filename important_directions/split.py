import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from important_directions.datasets.debug_printing import get_line_info
from important_directions.datasets.transformed_sklearn_dm import TransformedSklearnDataModule


def get_regression_split(X, y, test_size=0.2, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    #print(f"{y_train.mean()=}, {y_test.mean()=}")

    return X_train, X_test, y_train, y_test, x_scaler, y_scaler


def get_dataset_ptl(X, y, batch_size=64, scale_x=True, scale_y=True, *args, **kwargs):
    x_tr = StandardScaler if scale_x else None
    y_tr = StandardScaler if scale_y else None

    #print(f"{get_line_info()} {y.dtype=}")

    data_module = TransformedSklearnDataModule(X=X.astype(np.float32), y=y.astype(np.float32), batch_size=batch_size,
                                               X_transform_cls=x_tr, y_transform_cls=y_tr, *args, **kwargs)

    return data_module
