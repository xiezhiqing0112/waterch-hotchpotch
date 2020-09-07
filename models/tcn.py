from collections import OrderedDict
from typing import List

import torch
from torch import nn


class ChompLayer(nn.Module):
    def __init__(self, chomp_size: int):
        super(ChompLayer, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
            self,
            in_channels: int, out_channels: int, kernel_size: int,
            stride: int, dilation: int, padding, dropout=0.2, activation=nn.ReLU
    ):
        super(TemporalBlock, self).__init__()

        self.add_module('casual', nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )),
            ChompLayer(padding),
            nn.BatchNorm1d(out_channels),
            activation(True),
            nn.Dropout(dropout) if dropout >= 1e-6 else nn.Sequential(),
            nn.utils.weight_norm(nn.Conv1d(
                out_channels, out_channels, kernel_size,
                stride=stride, padding=padding, dilation=dilation
            )),
            ChompLayer(padding),
            nn.BatchNorm1d(out_channels),
            activation(True),
            nn.Dropout(dropout)
        ))
        self.add_module('down_sample', nn.Conv1d(
            in_channels, out_channels, 1
        ) if in_channels != out_channels else nn.Sequential())
        self.add_module('activation', activation(True))

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                module.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor):
        children = OrderedDict(self.named_children())
        return children['activation'](
            children['casual'](x) + children['down_sample'](x)
        )


class TemporalConvNet(nn.Sequential):
    def __init__(self, in_channels: int, layer_channels: List[int], kernel_size: int, dropout: float=0.2, activation=nn.ReLU, **kwargs):
        def build_layer(index: int):
            dilation = 2 ** index
            in_channels_ = in_channels if index == 0 else layer_channels[index - 1]
            out_channels = layer_channels[index]
            return TemporalBlock(
                in_channels_, out_channels, kernel_size,
                stride=1, dilation=dilation, padding=(kernel_size - 1) * dilation,
                dropout=dropout, activation=activation
            )

        super(TemporalConvNet, self).__init__(
            *tuple(map(
                build_layer,
                range(len(layer_channels))
            ))
        )