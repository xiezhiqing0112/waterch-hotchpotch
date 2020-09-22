__all__ = [
    'NAIC2020RSModelTrainTask'
]

from logging import Logger, getLogger as get_logger
from typing import OrderedDict

import torch
from tasker import Profile
from tasker.storage import Storage
from tasker.tasks.torch import SimpleTrainTask
from torch import nn
from ignite import metrics
from torchvision.models.segmentation import deeplabv3_resnet50


class NAIC2020RSModelTrainTask(SimpleTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        class Wrapper(nn.Module):
            def __init__(self, **kwargs):
                super(Wrapper, self).__init__()
                self.add_module('deeplabv3', deeplabv3_resnet50(pretrained=False, num_classes=8))

            def forward(self, x: torch.Tensor):
                y: torch.Tensor = self.deeplabv3(x)
                return y['out']

        return Wrapper()

    @classmethod
    def define_model(cls):
        return []

    def more_metrics(self, metrics_: OrderedDict):
        metrics_['loss'] = metrics.Loss(nn.CrossEntropyLoss())
        metrics_['accuracy'] = metrics.Accuracy()
        metrics_['confusion_matrix'] = metrics.ConfusionMatrix(8)
        metrics_['iou'] = metrics.IoU(metrics_['confusion_matrix'])
