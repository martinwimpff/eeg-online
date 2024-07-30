from eeg_online.models import BaseNet


model_dict = dict(BaseNet=BaseNet)


def get_model_cls(model_name: str):
    return model_dict.get(model_name, NotImplementedError)
