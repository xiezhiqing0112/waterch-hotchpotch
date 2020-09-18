__all__ = [
    'NAIC2020RSModelTrainTask'
]

from logging import Logger
from typing import OrderedDict

from tasker import Profile
from tasker.storage import Storage
from tasker.tasks.torch import SimpleTrainTask
from torch import nn
from ignite import metrics


class NAIC2020RSModelTrainTask(SimpleTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        class Model(nn.Module):
            pass
        return Model()

    @classmethod
    def define_model(cls):
        return []

    def more_metrics(self, metrics_: OrderedDict):
        metrics_['loss'] = metrics.Loss(nn.CrossEntropyLoss())
        metrics_['accuracy'] = metrics.Accuracy()
        metrics_['confusion_matrix'] = metrics.ConfusionMatrix(8)
        metrics_['iou'] = metrics.IoU(metrics_['confusion_matrix'])