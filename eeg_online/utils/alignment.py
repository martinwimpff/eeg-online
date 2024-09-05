import numpy as np
from pyriemann.utils.mean import mean_riemann
from scipy import linalg


def _alignment(method: str, x: np.ndarray, x_test: np.ndarray = None,
               return_cov: bool = False, n_windows_per_trial: int = 61):
    trial_covs = []
    for trial in x:
        windows = []
        for i in range(n_windows_per_trial):
            windows.append(trial[:, i * 16:i * 16 + 256])
        windows = np.array(windows)
        window_covmats = np.matmul(windows, windows.transpose((0, 2, 1)))
        if method == "euclidean":
            trial_covs.append(window_covmats.mean(0))
        elif method == "riemann":
            trial_covs.append(mean_riemann(window_covmats).astype("float32"))
        else:
            raise NotImplementedError

    if method == "euclidean":
        R = np.array(trial_covs).mean(0)
    elif method == "riemann":
        R = mean_riemann(np.array(trial_covs)).astype("float32")
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
