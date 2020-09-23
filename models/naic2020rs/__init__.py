__all__ = [
    'NAIC2020RSModelTrainTask'
]

from logging import Logger, getLogger as get_logger
from typing import OrderedDict, Dict, Text, Any, List

import torch
from tasker import Profile
from tasker.storage import Storage
from tasker.tasks.torch import SimpleTrainTask
from torch import nn
from ignite import metrics, engine
from torchvision.models.segmentation import deeplabv3_resnet50


class NAIC2020RSModelTrainTask(SimpleTrainTask):
    def __init__(self, prefix: Text = None):
        super(NAIC2020RSModelTrainTask, self).__init__(prefix)
        if prefix is not None:
            self.PROVIDE_STATE_KEY = f'{prefix}_train_state'
        else:
            self.PROVIDE_STATE_KEY = 'train_state'

    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        class Wrapper(nn.Module):
            def __init__(self, **kwargs):
                super(Wrapper, self).__init__()
                self.add_module('deeplabv3', nn.DataParallel(deeplabv3_resnet50(pretrained=False, num_classes=8)))

            def forward(self, x: torch.Tensor):
                y: torch.Tensor = self.deeplabv3(x)
                return y['out']

        return Wrapper()

    def on_epoch_completed(
            self,
            engine_: engine.Engine, metrics_: Dict[Text, Any],
            profile: Profile, shared: Storage, logger: Logger
    ):
        super(NAIC2020RSModelTrainTask, self).on_epoch_completed(engine_, metrics_, profile, shared, logger)
        if self.PROVIDE_STATE_KEY in shared:
            state_list: List[Dict] = shared[self.PROVIDE_STATE_KEY]
            state_list.append(engine_.state_dict())
            shared[self.PROVIDE_STATE_KEY] = state_list
        else:
            shared[self.PROVIDE_STATE_KEY] = [engine_.state_dict()]
        logger.info(f'Dumped state of epoch {engine_.state.epoch}')

    @classmethod
    def define_model(cls):
        return []

    def provide(self) -> List[Text]:
        provide_list = super(NAIC2020RSModelTrainTask, self).provide()
        provide_list.append(self.PROVIDE_STATE_KEY)
        return provide_list

    def require(self) -> List[Text]:
        require_list = super(NAIC2020RSModelTrainTask, self).require()
        require_list.append(self.PROVIDE_STATE_KEY)
        return require_list

    def more_metrics(self, metrics_: OrderedDict):
        metrics_['loss'] = metrics.Loss(nn.CrossEntropyLoss())
        metrics_['accuracy'] = metrics.Accuracy()
        metrics_['confusion_matrix'] = metrics.ConfusionMatrix(8)
        metrics_['iou'] = metrics.IoU(metrics_['confusion_matrix'])
