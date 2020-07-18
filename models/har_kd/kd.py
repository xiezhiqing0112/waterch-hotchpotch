from collections import OrderedDict
from logging import Logger
from typing import List, Text

from ignite import metrics
from ignite.engine import Engine
from torch import nn
from waterch.tasker import Profile, value
from waterch.tasker.storage import Storage
from waterch.tasker.tasks.torch import SimpleTrainTask

from loss_functions.kd import HintonCrossEntropyLoss
from models.resnet1d import ResNet1d


class ResNetTeacherTrainTask(SimpleTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        return ResNet1d(**profile)

    @classmethod
    def define_model(cls):
        return ResNet1d.define()

    def more_metrics(self, metrics_: OrderedDict):
        metrics_['loss'] = metrics.Loss(nn.CrossEntropyLoss())
        metrics_['accuracy'] = metrics.Accuracy()
        metrics_['recall'] = metrics.Recall()
        metrics_['precision'] = metrics.Precision()


class ResNetStudentTrainTask(ResNetTeacherTrainTask):
    def create_trainer(
            self, model, optimizer, loss_fn, device, non_blocking, prepare_batch,
            output_transform=lambda x, y, y_pred, loss: loss.item()
    ):
        if device:
            model.to(device)

        def _update(engine, batch):
            model.train()
            optimizer.zero_grad()
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            loss = loss_fn(x, y_pred, y)
            loss.backward()
            optimizer.step()
            return output_transform(x, y, y_pred, loss)

        return Engine(_update)

    def create_loss(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        return HintonCrossEntropyLoss(shared['teacher_model'], **profile)

    @classmethod
    def define_loss(cls):
        return [
            value('ratio', float),
            value('temperature', float)
        ]

    def require(self) -> List[Text]:
        require_keys = super(ResNetStudentTrainTask, self).require()
        require_keys += ['teacher_model']
        return require_keys
