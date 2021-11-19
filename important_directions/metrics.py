import numpy as np


def get_cov_prob(y_pred, y_lower, y_upper, y_true, rtol=1.e-4, atol=1.e-6):
    """
    Calculate coverage probability given true output, predicted mean and confidence limits.
    Additionally check if prediction is very close to the true output.
    """
    n_cov_new = (((y_true >= y_lower) & (y_true <= y_upper))
                 | np.isclose(y_pred, y_true, rtol=rtol, atol=atol)) \
        .astype(float)
    # print(n_cov_new)
    p_cov = np.mean(n_cov_new)
    return n_cov_new.sum().astype(int), p_cov
