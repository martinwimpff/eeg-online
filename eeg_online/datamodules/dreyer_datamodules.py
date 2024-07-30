from torch.utils.data import DataLoader

from .base_datamodule import BaseDataModuleLMSO, BaseDataModuleWithin
from .dreyer_datasets import DreyerDatasetLMSO, DreyerDatasetSingleSubject
from .window_dataset import WindowDataSet


class DreyerDataModuleLMSO(BaseDataModuleLMSO):
    def __init__(self, n_subjects: int = 79, n_folds: int = 79,
                 preprocessing_dict: dict = None):
        super(DreyerDataModuleLMSO, self).__init__(
            n_subjects, n_folds, preprocessing_dict)
        self.dataset = DreyerDatasetLMSO(n_subjects=n_subjects, n_folds=n_folds,
                                         **preprocessing_dict)


class DreyerDataModuleWithin(BaseDataModuleWithin):
    def __init__(self, n_subjects: int = 79, n_folds: int = 79,
                 preprocessing_dict: dict = None):
        super(DreyerDataModuleWithin, self).__init__(
            n_subjects, n_folds, preprocessing_dict)
        self.dataset = DreyerDatasetSingleSubject(n_subjects, preprocessing_dict)


class DreyerDataModuleTTA(DreyerDataModuleWithin):
    def __init__(self, n_subjects: int = 79, n_folds: int = 79,
                 preprocessing_dict: dict = None):
        super(DreyerDataModuleTTA, self).__init__(n_subjects, n_folds,
                                                  preprocessing_dict)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(WindowDataSet(self.test_dataset, self.dataset.n_windows_per_trial),
                          batch_size=1, shuffle=False)
