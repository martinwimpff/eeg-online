import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from .window_dataset import WindowDataSet


class BaseDataModuleLMSO(pl.LightningDataModule):
    dataset = None
    train_dataset = None
    cal_dataset = None
    test_dataset = None

    def __init__(self, n_subjects: int = 79, n_folds: int = 79,
                 preprocessing_dict: dict = None):
        super(BaseDataModuleLMSO, self).__init__()
        self.batch_size = preprocessing_dict.pop("batch_size", 64)

    def setup_fold(self, fold_idx: int):
        x_train, y_train, x_cal, y_cal, x_test, y_test = self.dataset.setup_fold(
            fold_idx)
        self.train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        self.cal_dataset = TensorDataset(torch.tensor(x_cal), torch.tensor(y_cal))
        self.test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.cal_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(WindowDataSet(self.test_dataset, self.dataset.n_windows_per_trial),
                          batch_size=self.dataset.n_windows_per_trial, shuffle=False)

    def get_test_subject_ids(self, fold_idx: int = 0):
        return self.dataset.all_subject_ids[self.dataset.splits[fold_idx][-1]]


class BaseDataModuleWithin(pl.LightningDataModule):
    dataset = None
    train_dataset = None
    test_dataset = None

    def __init__(self, n_subjects: int = 79, n_folds: int = 79,
                 preprocessing_dict: dict = None):
        super(BaseDataModuleWithin, self).__init__()
        assert n_subjects == n_folds
        self.n_subjects = n_subjects
        self.batch_size = preprocessing_dict.pop("batch_size", 64)
        self.preprocessing_dict = preprocessing_dict

    def setup_fold(self, fold_idx: int = 0):
        x_train, y_train, x_test, y_test = self.dataset.setup_fold(
            fold_idx)
        self.train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
        self.test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(WindowDataSet(self.test_dataset, self.dataset.n_windows_per_trial),
                          batch_size=self.dataset.n_windows_per_trial, shuffle=False)

    def get_test_subject_ids(self, fold_idx: int = 0):
        return [self.dataset.all_subject_ids[fold_idx]]
