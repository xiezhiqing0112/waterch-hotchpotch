from logging import Logger

import torch
from tasker import Profile
from tasker.storage import Storage
from tasker.tasks.torch import SimpleTrainTask
from torch import nn

from models.hopfield import HopfieldCore
from models.resnet1d import ResNet1d


class HARKDTargetModelTrainTask(SimpleTrainTask):
    class Model(nn.Module):
        def __init__(self, **kwargs):
            resnet_kwargs = kwargs['resnet']
            HopfieldCore
            pass

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            pass

    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        return self.Model(**profile)

    @classmethod
    def define_model(cls):
        pass
