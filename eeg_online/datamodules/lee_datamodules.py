from torch.utils.data import DataLoader

from .base_datamodule import BaseDataModuleLMSO, BaseDataModuleWithin
from .lee_datasets import LeeDatasetLMSO, LeeDatasetSingleSubject
from .window_dataset import WindowDataSet


class LeeDataModuleLMSO(BaseDataModuleLMSO):
    def __init__(self, n_subjects: int = 54, n_folds: int = 54,
                 preprocessing_dict: dict = None):
        super(LeeDataModuleLMSO, self).__init__(
            n_subjects, n_folds, preprocessing_dict)
        self.dataset = LeeDatasetLMSO(n_subjects=n_subjects, n_folds=n_folds,
                                      **preprocessing_dict)


class LeeDataModuleWithin(BaseDataModuleWithin):
    def __init__(self, n_subjects: int = 54, n_folds: int = 54,
                 preprocessing_dict: dict = None):
        super(LeeDataModuleWithin, self).__init__(
            n_subjects, n_folds, preprocessing_dict)
        self.dataset = LeeDatasetSingleSubject(n_subjects, preprocessing_dict)


class LeeDataModuleTTA(LeeDataModuleWithin):
    def __init__(self, n_subjects: int = 79, n_folds: int = 79,
                 preprocessing_dict: dict = None):
        super(LeeDataModuleTTA, self).__init__(n_subjects, n_folds,
                                                  preprocessing_dict)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(WindowDataSet(self.test_dataset, self.dataset.n_windows_per_trial),
                          batch_size=1, shuffle=False)
