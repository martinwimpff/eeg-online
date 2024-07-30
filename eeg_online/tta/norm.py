import torch
from torch import nn

from .alignment import OnlineAlignment
from .base import TTAMethod
from .bn import RobustBN


class Norm(TTAMethod):
    def __init__(self, model: nn.Module, config: dict):
        super(Norm, self).__init__(model, config)
        if self.config.get("alignment", False):
            self.reference = None
            self.counter = 0

    def forward_sliding_window(self, x):
        return self.forward_and_adapt(x)

    @torch.no_grad()
    def forward_and_adapt(self, x):
        if self.config.get("alignment", False):
            x = OnlineAlignment.align_data(self, x, self.config.get("alignment"))
        outputs = self.model(x)
        return outputs

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

        self.model = RobustBN.adapt_model(
            self.model, alpha=self.config.get("alpha"))
