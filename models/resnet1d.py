from typing import List

import torch
from torch import nn
from waterch.tasker import Definition, value
from waterch.tasker.mixin import ProfileMixin


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(BasicBlock1d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        if 'activation' in kwargs:
            activation = kwargs['activation']
        else:
            activation = nn.ReLU

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = activation(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck1d(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(Bottleneck1d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if 'activation' in kwargs:
            activation = kwargs['activation']
        else:
            activation = nn.ReLU

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = activation(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet1d(nn.Module, ProfileMixin):

    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('in_channels', int),
            value('block', str),
            value('layers', list, [int]),
            value('channels', list, [int]),
            value('num_classes', int),
            value('zero_init_residual', bool),
            value('groups', int),
            value('width_per_group', int),
            value('enable_classifier', bool)
        ]

    def __init__(self, in_channels, block, layers, channels, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, **kwargs):
        super(ResNet1d, self).__init__()
        replace_stride_with_dilation = kwargs['replace_stride_with_dilation'] \
            if 'replace_stride_with_dilation' in kwargs else None
        norm_layer = kwargs['norm_layer'] if 'norm_layer' in kwargs else None
        self.enable_classifier = kwargs['enable_classifier'] if 'enable_classifier' in kwargs else True

        assert len(layers) == len(channels)

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer
        assert block in ('basic_block', 'bottleneck')
        if block == 'bottleneck':
            block = Bottleneck1d
        else:
            block = BasicBlock1d
        if 'activation' in kwargs:
            activation = kwargs['activation']
        else:
            activation = nn.ReLU

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False for idx in range(len(channels) - 1)]
        assert len(replace_stride_with_dilation) == len(channels) - 1
        if channels is None:
            channels = [64, 128, 256, 512]

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = activation(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.Sequential(*tuple(map(
            lambda idx: self._make_layer(block, channels[idx], layers[idx])
            if idx == 0
            else self._make_layer(
                block, channels[idx], layers[idx],
                stride=2, dilate=replace_stride_with_dilation[idx - 1]
            ),
            range(len(layers))
        )))
        if self.enable_classifier:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(channels[-1] * block.expansion, num_classes)
        else:
            self.fc = nn.Conv1d(in_channels=channels[-1] * block.expansion, out_channels=channels[-1], kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck1d):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock1d):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        if 'activation' in kwargs:
            activation = kwargs['activation']
        else:
            activation = nn.ReLU

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, activation=activation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layers(x)

        if self.enable_classifier:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
