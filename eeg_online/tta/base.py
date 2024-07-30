import torch
from torch import nn


class TTAMethod(nn.Module):
    def __init__(self, model: nn.Module, config: dict):
        super(TTAMethod, self).__init__()
        self.model = model
        self.config = config
        self.device = self.model.device

        self.configure_model()

    def forward(self, x):
        assert x.shape[0] == 1  # Only single-sample test-time adaptation allowed
        outputs = self.forward_and_adapt(x)

        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def forward_sliding_window(self, x):
        return self.model(x)

    def configure_model(self):
        raise NotImplementedError

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names
