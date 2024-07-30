from torch import nn
from torch.utils.data import TensorDataset


def reset_bn(model: nn.Module, dataset: TensorDataset):
    model.eval()  # deactivate dropout
    momentums = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()  # delete train statistics
            momentums.append(m.momentum)
            m.momentum = 1  # no running mean
            m.train()

    x, _ = dataset.tensors
    device = model.device
    _ = model(x.to(device))

    # reset momentum
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            momentums.append(m.momentum)
            m.momentum = momentums.pop(0)

    return model
