import numpy as np

from .base_dataset import BaseDataset, BaseDatasetSingleSubject
from eeg_online.utils.load_data_lee import load_run


class LeeDatasetLMSO(BaseDataset):
    all_subject_ids = np.array([str(i) for i in range(1, 55)])

    def __init__(self, n_subjects: int = 54, n_folds: int = 54, l_freq: int = 8,
                 h_freq: int = 30, baseline_correction: bool = False,
                 tmin: float = 0.25, sfreq: int = 256,
                 alignment: str | bool | None = False):
        super(LeeDatasetLMSO, self).__init__(
            n_subjects=n_subjects, n_folds=n_folds, tmin=tmin, alignment=alignment)
        self.n_windows_per_trial = int(16 * (4 - self.tmin - 1)) + 1
        self.train_run_ids = [1]
        self.cal_run_ids = [1]
        self.test_run_ids = [2]
        for subject_id in self.all_subject_ids:
            data = [load_run(int(subject_id), run_id, l_freq, h_freq,
                             baseline_correction, sfreq) for run_id in range(1, 3)]
            self.data_dict.update({subject_id: {
                "data": {
                    f"run_{i+1}": epochs.get_data(tmin=tmin, tmax=4.0) for i, (epochs, _) in enumerate(data)},
                "labels": {
                    f"run_{i+1}": labels for i, (_, labels) in enumerate(data)},
            }})


class LeeDatasetSingleSubject(BaseDatasetSingleSubject):
    all_subject_ids = np.array([str(i) for i in range(1, 55)])

    def __init__(self, n_subjects: int, preprocessing_dict: dict):
        super(LeeDatasetSingleSubject, self).__init__(n_subjects, preprocessing_dict)
        self.n_windows_per_trial = int(16 * (4 - self.tmin - 1)) + 1
        self.train_run_ids = [1]
        self.test_run_ids = [2]
        self.preprocessing_dict = preprocessing_dict

    def setup_fold(self, fold_idx: int = 0):
        return super(LeeDatasetSingleSubject, self)._setup_fold(
            load_run, n_runs=2, tmax=4.0, fold_idx=fold_idx)
