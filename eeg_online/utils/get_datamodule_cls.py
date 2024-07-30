from eeg_online.datamodules import DreyerDataModuleLMSO, \
    DreyerDataModuleWithin, LeeDataModuleLMSO, LeeDataModuleWithin


dm_dict = dict(
    DreyerDataModuleLMSO=DreyerDataModuleLMSO,
    DreyerDataModuleWithin=DreyerDataModuleWithin,
    LeeDataModuleLMSO=LeeDataModuleLMSO,
    LeeDataModuleWithin=LeeDataModuleWithin
)


def get_datamodule_cls(datamodule_name: str):
    return dm_dict.get(datamodule_name, NotImplementedError)
