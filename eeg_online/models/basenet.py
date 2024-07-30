from einops.layers.torch import Rearrange
import torch
from torch import nn

from .classification_module import ClassificationModule


class BaseNetModule(nn.Module):
    def __init__(self,
                 n_channels: int = 27,
                 n_temporal_filters: int = 40,
                 temp_filter_length_inp: int = 25,
                 spatial_expansion: int = 1,
                 pool_length_inp: int = 2,
                 dropout_inp: int = 0.5,
                 ch_dim: int = 16,
                 temp_filter_length: int = 15,
                 dropout: float = 0.5,
                 padding_mode: str = "zeros"):
        super(BaseNetModule, self).__init__()
        assert pool_length_inp in [1, 2, 4, 8]
        self.input_block = nn.Sequential(
            Rearrange("b c t -> b 1 c t"),
            nn.Conv2d(
                1, n_temporal_filters, (1, temp_filter_length_inp),
                padding=(0, temp_filter_length_inp // 2), bias=False,
                padding_mode=padding_mode
            ),
            nn.BatchNorm2d(n_temporal_filters),
            nn.Conv2d(
                n_temporal_filters, n_temporal_filters * spatial_expansion,
                (n_channels, 1), groups=n_temporal_filters, bias=False),
            nn.BatchNorm2d(n_temporal_filters * spatial_expansion),
            nn.ELU(),
            nn.AvgPool2d((1, pool_length_inp), (1, pool_length_inp)),
            nn.Dropout(dropout_inp),
        )

        self.channel_expansion = nn.Sequential(
            nn.Conv2d(
                n_temporal_filters * spatial_expansion, ch_dim, (1, 1), bias=False),
            nn.BatchNorm2d(ch_dim),
            nn.ELU()
        )

        pool_length, pool_stride = int(256 / pool_length_inp), int(16 / pool_length_inp)
        self.fe = nn.Sequential(
            nn.Conv2d(ch_dim, ch_dim, (1, temp_filter_length),
                      padding=(0, temp_filter_length // 2), bias=False, groups=ch_dim,
                      padding_mode=padding_mode),
            nn.Conv2d(ch_dim, ch_dim, (1, 1), bias=False),
            nn.BatchNorm2d(ch_dim),
            nn.ELU(),
            nn.AvgPool2d((1, pool_length), (1, pool_stride)),
            nn.Dropout(dropout),
            Rearrange("b f 1 t -> b t f")
        )

        self.classifier = nn.Sequential(
            nn.Linear(ch_dim, 1),
            Rearrange("b t 1 -> b t")
        )

    def forward(self, x: torch.tensor):
        x = self.input_block(x)
        x = self.channel_expansion(x)
        x = self.fe(x)
        return self.classifier(x)


class BaseNet(ClassificationModule):
    def __init__(self,
                 n_channels: int = 27,
                 n_temporal_filters: int = 40,
                 temp_filter_length_inp: int = 25,
                 spatial_expansion: int = 1,
                 pool_length_inp: int = 2,
                 dropout_inp: int = 0.5,
                 ch_dim: int = 16,
                 temp_filter_length: int = 15,
                 dropout: float = 0.5,
                 padding_mode: str = "zeros",
                 **kwargs):
        model = BaseNetModule(
            n_channels=n_channels,
            n_temporal_filters=n_temporal_filters,
            temp_filter_length_inp=temp_filter_length_inp,
            spatial_expansion=spatial_expansion,
            pool_length_inp=pool_length_inp,
            dropout_inp=dropout_inp,
            ch_dim=ch_dim,
            temp_filter_length=temp_filter_length,
            dropout=dropout,
            padding_mode=padding_mode
        )
        super(BaseNet, self).__init__(model, **kwargs)
