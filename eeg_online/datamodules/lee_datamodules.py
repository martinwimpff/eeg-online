from .base_datamodule import BaseDataModuleLMSO, BaseDataModuleWithin
from .lee_datasets import LeeDatasetLMSO, LeeDatasetSingleSubject


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
