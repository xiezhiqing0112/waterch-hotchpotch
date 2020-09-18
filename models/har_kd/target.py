from logging import Logger

import torch
from tasker import Profile
from tasker.storage import Storage
from tasker.tasks.torch import SimpleTrainTask
from torch import nn

from models.hopfield import HopfieldCore


class HARKDTargetModelTrainTask(SimpleTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        class TargetModel(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()

            def forward(self, x: torch.Tensor):
                pass

        return TargetModel()

    @classmethod
    def define_model(cls):
        pass
