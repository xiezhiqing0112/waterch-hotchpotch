import torch
from torch import nn
from math import floor
from multiprocessing.dummy import Pool

from typing import AnyStr, List, Dict, Text, Any

from models.attention import SqueezeExcitationBlock


class DenseBlockHiddenBasic(nn.Module):
    @staticmethod
    def _padding_size(kernel_size):
        return int(floor((kernel_size - 1) / 2))

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout=0.0):
        super(DenseBlockHiddenBasic, self).__init__()
        self.add_module('batch_norm', nn.BatchNorm1d(in_channels))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
            stride=1,
            padding=self._padding_size(kernel_size)
        ))
        if 0.0 < dropout <= 1.0:
            self.add_module('dropout', nn.Dropout(dropout))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        children = {name: module for name, module in self.named_children()}
        if 'dropout' in children:
            return children['dropout'](children['conv'](children['relu'](children['batch_norm'](input))))
        else:
            return children['conv'](children['relu'](children['batch_norm'](input)))


class DenseBlockHiddenBottleneck(DenseBlockHiddenBasic):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout=0.0):
        super(DenseBlockHiddenBottleneck, self).__init__(in_channels * 4, out_channels, kernel_size, dropout)
        self.add_module('batch_norm_b', nn.BatchNorm1d(in_channels))
        self.add_module('relu_b', nn.ReLU())
        self.add_module('conv_b', nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * 4,
            kernel_size=kernel_size,
            bias=False,
            stride=1,
            padding=0
        ))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        children = {name: module for name, module in self.named_children()}
        return super(DenseBlockHiddenBottleneck, self).forward(
            children['conv_b'](children['relu_b'](children['batch_norm_b'](input)))
        )


class DenseBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            growth_rate: int,
            kernel_size: int,
            hidden_layers: int,
            block_type: AnyStr = 'bottleneck',
            dropout: float = 0.0
    ):
        super(DenseBlock, self).__init__()
        self._hidden_layers = hidden_layers
        for index in range(self._hidden_layers):
            self.add_module('hidden_{0}'.format(index), DenseBlockHiddenBottleneck(
                in_channels=in_channels + index * growth_rate,
                out_channels=growth_rate,
                kernel_size=kernel_size,
                dropout=dropout
            ) if block_type == 'bottleneck' else DenseBlockHiddenBasic(
                in_channels=in_channels + index * growth_rate,
                out_channels=growth_rate,
                kernel_size=kernel_size,
                dropout=dropout
            ))
        self.out_channels = in_channels + (index + 1) * growth_rate

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        children = {name: module for name, module in self.named_children()}
        hidden_input = input
        for index in range(self._hidden_layers):
            output = children['hidden_{0}'.format(index)](hidden_input)
            hidden_input = torch.cat((hidden_input, output), 1)
        return hidden_input

    @property
    def hidden_layers(self):
        return self._hidden_layers


class TransitionLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            compression: float,
            pool_kernel_size: int = 2,
            pool_stride: int = 2,
            dropout: float = 0.0
    ):
        out_channels = in_channels
        if 0.0 <= compression < 1.0:
            out_channels = int(floor(compression * in_channels))
        super(TransitionLayer, self).__init__()
        self.add_module('batch_norm', nn.BatchNorm1d(in_channels))
        self.add_module('relu', nn.ReLU())
        self.add_module('conv', nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        ))
        if 0.0 < dropout <= 1.0:
            self.add_module('dropout', nn.Dropout(dropout))
        self.add_module('pool', nn.AvgPool1d(
            kernel_size=pool_kernel_size,
            stride=pool_stride
        ))

    def forward(self, input):
        children = {name: module for name, module in self.named_children()}
        output = children['conv'](children['relu'](children['batch_norm'](input)))
        if 'dropout' in children:
            output = children['dropout'](output)
        output = children['pool'](output)
        return output


class DenseNet(nn.Module):
    def __init__(
            self,
            in_channels: int,
            growth_rate: int,
            block_number: int,
            input_conv_kernel_size: int,
            input_conv_stride: int,
            input_conv_padding: int,
            input_pool_kernel_size: int,
            input_pool_stride: int,
            input_pool_padding: int,
            block_kernel_sizes: List[int],
            block_hidden_layers: List[int],
            transition_pool_kernel_sizes: List[int],
            transition_pool_strides: List[int],
            bottleneck: bool = True,
            compression: float = 1.0,
            dropout: float = 0.0
    ):
        super(DenseNet, self).__init__()
        out_channels = 2 * growth_rate
        self.add_module('conv_input', nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=input_conv_kernel_size,
            stride=input_conv_stride,
            padding=input_conv_padding
        ))
        self.add_module('pool_input', nn.AvgPool1d(
            kernel_size=input_pool_kernel_size,
            stride=input_pool_stride,
            padding=input_pool_padding
        ))
        self._block_number = block_number
        in_channels = out_channels
        for index in range(self._block_number):
            if index > 0:
                self.add_module('transition_{0}'.format(index - 1), TransitionLayer(
                    in_channels=in_channels,
                    compression=compression,
                    pool_kernel_size=transition_pool_kernel_sizes[index - 1],
                    pool_stride=transition_pool_strides[index - 1],
                    dropout=dropout
                ))
                in_channels = int(floor(compression * in_channels))
            self.add_module('block_{0}'.format(index), DenseBlock(
                in_channels=in_channels,
                growth_rate=growth_rate,
                kernel_size=block_kernel_sizes[index],
                hidden_layers=block_hidden_layers[index],
                block_type='bottleneck' if bottleneck else 'basic',
                dropout=dropout
            ))
            in_channels += block_hidden_layers[index] * growth_rate

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        children = {name: module for name, module in self.named_children()}
        call = ['conv_input', 'pool_input', 'block_0']
        for index in range(1, self._block_number):
            call.extend([
                'transition_{0}'.format(index - 1),
                'block_{0}'.format(index)
            ])
        output = input
        for name in call:
            output = children[name](output)
        return output


class SELayer(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: float):
        super(SELayer, self).__init__()

        self.add_module('se', SqueezeExcitationBlock(in_channels, reduction_ratio))

    def forward(self, *input: Any, **kwargs: Any) -> torch.Tensor:
        children = dict(self.named_children())

        return children['se'](input[0]).softmax(dim=1) * input[0]


class SEDenseNet(DenseNet):
    def __init__(self, *args, **kwargs):
        reduction_ratio = kwargs['reduction_ratio'] if 'reduction_ratio' in kwargs else 2.0
        if 'reduction_ratio' in kwargs:
            kwargs.pop('reduction_ratio')

        super(SEDenseNet, self).__init__(*args, **kwargs)
        children = dict(self.named_children())

        self.add_module('se_input', SELayer(
            in_channels=children['conv_input'].out_channels,
            reduction_ratio=reduction_ratio
        ))

        for index in range(self._block_number):
            self.add_module(f'se_{index}', SELayer(
                children[f'block_{index}'].out_channels,
                reduction_ratio=reduction_ratio
            ))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        children = {name: module for name, module in self.named_children()}
        call = ['conv_input', 'pool_input', 'se_input', 'block_0', 'se_0']
        for index in range(1, self._block_number):
            call.extend([
                'transition_{0}'.format(index - 1),
                'block_{0}'.format(index),
                f'se_{index}'
            ])
        output = input
        for name in call:
            output = children[name](output)
        return output


class DenseNetLinear(nn.Module):
    pool = Pool(10)

    def __init__(self, config: Dict[Text, Any]):
        super(DenseNetLinear, self).__init__()

        self.add_module('densenet', nn.ModuleList(map(
            lambda subnet: nn.DataParallel(DenseNet(
                in_channels=subnet['in_channels'],
                growth_rate=subnet['growth_rate'],
                block_number=subnet['block_number'],
                input_conv_kernel_size=subnet['input']['conv_kernel_size'],
                input_conv_stride=subnet['input']['conv_stride'] if 'conv_stride' in subnet['input'] else 1,
                input_conv_padding=subnet['input']['conv_padding'],
                input_pool_kernel_size=subnet['input']['pool_kernel_size'],
                input_pool_stride=subnet['input']['pool_stride'] if 'pool_stride' in subnet['input'] else 1,
                input_pool_padding=subnet['input']['pool_padding'],
                block_kernel_sizes=subnet['block']['kernel_sizes'],
                block_hidden_layers=subnet['block']['hidden_layers'],
                transition_pool_kernel_sizes=subnet['transition']['pool_kernel_sizes'],
                transition_pool_strides=subnet['transition']['pool_strides'],
                bottleneck=subnet['bottleneck'] if 'bottleneck' in subnet else True,
                compression=subnet['compression'] if 'compression' in subnet else 1.0,
                dropout=subnet['dropout'] if 'dropout' in subnet else 0.0,
            ).to('cuda:0')),
            config['densenet']
        )))
        self.add_module('adaptive', nn.AdaptiveMaxPool1d(config['adaptive_size']))
        self.add_module('linear', nn.Linear(config['adaptive_size'], config['class_number']))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        children = dict(self.named_children())

        # concat = torch.cat(tuple(self.pool.map(
        concat = torch.cat(tuple(map(
            lambda tensor: tensor.reshape(tensor.shape[0], 1, -1),
            map(
                lambda index: children['densenet'][index](input[index]),
                range(len(input))
            )
        )), dim=2)
        return children['linear'](children['adaptive'](concat).reshape(concat.shape[0], -1))


class SEDenseNetLinear(DenseNetLinear):
    def __init__(self, config: Dict[Text, Any]):
        super(DenseNetLinear, self).__init__()

        self.add_module('densenet', nn.ModuleList(map(
            lambda subnet: SEDenseNet(
                in_channels=subnet['in_channels'],
                growth_rate=subnet['growth_rate'],
                block_number=subnet['block_number'],
                input_conv_kernel_size=subnet['input']['conv_kernel_size'],
                input_conv_stride=subnet['input']['conv_stride'] if 'conv_stride' in subnet['input'] else 1,
                input_conv_padding=subnet['input']['conv_padding'],
                input_pool_kernel_size=subnet['input']['pool_kernel_size'],
                input_pool_stride=subnet['input']['pool_stride'] if 'pool_stride' in subnet['input'] else 1,
                input_pool_padding=subnet['input']['pool_padding'],
                block_kernel_sizes=subnet['block']['kernel_sizes'],
                block_hidden_layers=subnet['block']['hidden_layers'],
                transition_pool_kernel_sizes=subnet['transition']['pool_kernel_sizes'],
                transition_pool_strides=subnet['transition']['pool_strides'],
                bottleneck=subnet['bottleneck'] if 'bottleneck' in subnet else True,
                compression=subnet['compression'] if 'compression' in subnet else 1.0,
                dropout=subnet['dropout'] if 'dropout' in subnet else 0.0,
                reduction_ratio=subnet['reduction_ratio'] if 'reduction_ratio' in subnet else 2.0
            ),
            config['densenet']
        )))
        self.add_module('adaptive', nn.AdaptiveMaxPool1d(config['adaptive_size']))
        self.add_module('linear', nn.Linear(config['adaptive_size'], config['class_number']))
