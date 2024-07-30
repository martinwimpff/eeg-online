from eeg_online.tta import Norm, OnlineAlignment


def get_tta_cls(tta_method: str):
    if tta_method == "norm":
        tta_cls = Norm
    elif tta_method == "alignment":
        tta_cls = OnlineAlignment
    else:
        raise NotImplementedError

    return tta_cls
