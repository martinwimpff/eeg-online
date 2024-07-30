from eeg_online.datamodules import DreyerDataModuleLMSO, DreyerDataModuleTTA, \
    DreyerDataModuleWithin, LeeDataModuleLMSO, LeeDataModuleTTA, LeeDataModuleWithin


dm_dict = dict(
    DreyerDataModuleLMSO=DreyerDataModuleLMSO,
    DreyerDataModuleTTA=DreyerDataModuleTTA,
    DreyerDataModuleWithin=DreyerDataModuleWithin,
    LeeDataModuleLMSO=LeeDataModuleLMSO,
    LeeDataModuleTTA=LeeDataModuleTTA,
    LeeDataModuleWithin=LeeDataModuleWithin
)


def get_datamodule_cls(datamodule_name: str):
    return dm_dict.get(datamodule_name, NotImplementedError)
