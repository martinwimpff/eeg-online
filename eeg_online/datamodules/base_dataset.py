from abc import ABC
from typing import Callable

import numpy as np
from sklearn.model_selection import KFold

from eeg_online.utils.alignment import align


class BaseDataset(ABC):
    """
    This is the Factory class for all datasets.
    Please set self.all_subject_ids, self.n_windows_per_trial, self.train_run_ids,
    self.cal_run_ids and self.test_run_ids in the init of your class.
    Also load the data in the data dict.
    """
    all_subject_ids = []
    n_windows_per_trial = None
    train_run_ids = []
    cal_run_ids = []
    test_run_ids = []
    data_dict = {}

    def __init__(self, n_subjects: int = 10, n_folds: int = 5, tmin: float = 0.25,
                 alignment: str | bool | None = False):
        self.tmin = tmin
        self.alignment = alignment
        kf = KFold(n_splits=n_folds, shuffle=False)
        self.all_subject_ids = self.all_subject_ids[:n_subjects]
        self.splits = [split for split in kf.split(self.all_subject_ids)]

    def setup_fold(self, fold_idx: int = 0):
        train_indices, test_indices = self.splits[fold_idx]

        x_train = np.concatenate([
            self._get_subject_data(self.all_subject_ids[idx], self.train_run_ids,
                                   "train")[0] for idx in train_indices])
        y_train = np.concatenate([
            self._get_subject_data(self.all_subject_ids[idx], self.train_run_ids,
                                   "train")[1] for idx in train_indices])
        x_cal = np.concatenate([self._get_subject_data(
            self.all_subject_ids[idx], self.cal_run_ids, "calibration")[0] for idx in
                                test_indices])
        y_cal = np.concatenate([self._get_subject_data(
            self.all_subject_ids[idx], self.cal_run_ids, "calibration")[1] for idx in
                                test_indices])
        x_test = np.concatenate([self._get_subject_data(
            self.all_subject_ids[idx], self.test_run_ids, "test")[0] for idx in
                                 test_indices])
        y_test = np.concatenate([self._get_subject_data(
            self.all_subject_ids[idx], self.test_run_ids, "test")[1] for idx in
                                 test_indices])

        return x_train, y_train, x_cal, y_cal, x_test, y_test

    def _get_subject_data(self, subject_id: str, run_ids: list, mode: str = "train"):
        x = np.concatenate([self.data_dict[subject_id]["data"][f"run_{i}"] for i in
                            run_ids], dtype="float32")
        y = np.concatenate([self.data_dict[subject_id]["labels"][f"run_{i}"] for i in
                            run_ids], dtype="int32")

        if mode == "train":
            if self.alignment not in [None, False]:
                R_op = self.data_dict[subject_id].get("R_op", None)
                if R_op is None:
                    x, R_op = align(self.alignment, x, return_cov=True,
                                    n_windows_per_trial=self.n_windows_per_trial)
                    self.data_dict[subject_id]["R_op"] = R_op
                else:
                    x = np.matmul(R_op, x)
        elif mode == "calibration":
            if self.alignment not in [None, False]:
                R_op_cal = self.data_dict[subject_id].get("R_op_cal", None)
                if R_op_cal is None:
                    x, R_op_cal = align(self.alignment, x, return_cov=True,
                                        n_windows_per_trial=self.n_windows_per_trial)
                    self.data_dict[subject_id]["R_op_cal"] = R_op_cal
                else:
                    x = np.matmul(R_op_cal, x)
        elif mode == "test":
            if self.alignment not in [None, False]:
                x = np.matmul(self.data_dict[subject_id]["R_op_cal"], x)
        else:
            raise NotImplementedError
        return x, y


class BaseDatasetSingleSubject(ABC):
    """
    This is the Factory class for all datasets.
    Please set self.all_subject_ids, self.n_windows_per_trial, self.train_run_ids and
    self.test_run_ids and the preprocessing dict in the init of your class.
    """
    all_subject_ids = []
    n_windows_per_trial = None
    train_run_ids = []
    test_run_ids = []
    R_op = None
    preprocessing_dict = None
    data_dict = None
    info = None

    def __init__(self, n_subjects: int, preprocessing_dict: dict):
        self.n_subjects = n_subjects
        self.tmin = preprocessing_dict.pop("tmin", 0.25)
        self.alignment = preprocessing_dict.pop("alignment", False)

    def _setup_fold(self, load_func: Callable, n_runs: int, tmax: float, fold_idx: int = 0):
        subject_id = self.all_subject_ids[fold_idx]
        data = [load_func(int(subject_id), run_id, **self.preprocessing_dict) for run_id
                in range(1, n_runs + 1)]
        self.data_dict = {
            subject_id: {
                "data": {
                    f"run_{i + 1}": epochs.get_data(tmin=self.tmin, tmax=tmax) for i, (epochs, _)
                    in enumerate(data)},
                "labels": {
                    f"run_{i + 1}": labels for i, (_, labels) in enumerate(data)}}
        }
        self.info = data[0][0].info

        x_cal, y_cal = self._get_data(subject_id, self.train_run_ids, True)
        x_test, y_test = self._get_data(subject_id, self.test_run_ids, False)
        return x_cal, y_cal, x_test, y_test

    def setup_fold(self, fold_idx: int = 0):
        """
        Use the _setup_fold function in your definition of the setup_fold function.
        """
        raise NotImplementedError

    def _get_data(self, subject_id: str, run_ids: list, train: bool = False):
        x = np.concatenate([self.data_dict[subject_id]["data"][f"run_{i}"] for i in
                            run_ids], dtype="float32")
        y = np.concatenate([self.data_dict[subject_id]["labels"][f"run_{i}"] for i in
                            run_ids], dtype="int32")
        if self.alignment not in [None, False]:
            if train:
                x, self.R_op = align(self.alignment, x, return_cov=True,
                                     n_windows_per_trial=self.n_windows_per_trial)
            else:
                x = np.matmul(self.R_op, x)
        return x, y
