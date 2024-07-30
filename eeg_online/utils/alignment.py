import numpy as np
from pyriemann.utils.mean import mean_riemann
from scipy import linalg


def _alignment(method: str, x: np.ndarray, x_test: np.ndarray = None,
               return_cov: bool = False, n_windows_per_trial: int = 61):
    if method == "euclidean":
        R = np.matmul(x, x.transpose((0, 2, 1))).mean(0)
    elif method == "riemann":
        covmats = np.matmul(x, x.transpose((0, 2, 1)))
        R = mean_riemann(covmats).astype("float32")
    else:
        raise NotImplementedError
    R_op = linalg.inv(linalg.sqrtm(R))
    x = np.matmul(R_op, x)
    if x_test is not None:
        x_test = np.matmul(R_op, x_test)
    if return_cov:
        return x, x_test, R_op
    else:
        return x, x_test


def align(method: str, x: np.ndarray, x_test: np.ndarray = None,
          return_cov: bool = False, n_windows_per_trial: int = 61):
    if return_cov:
        x, x_test, R_op = _alignment(method, x, x_test, return_cov, n_windows_per_trial)
        if x_test is not None:
            return x, x_test, R_op
        else:
            return x, R_op
    else:
        x, x_test = _alignment(method, x, x_test, return_cov, n_windows_per_trial)
        if x_test is not None:
            return x, x_test
        else:
            return x
