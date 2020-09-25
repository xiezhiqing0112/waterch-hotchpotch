import pickle
from datetime import datetime
from logging import Logger
from os import listdir
from pathlib import Path
from typing import OrderedDict, Dict, Text, Any, List
from zipfile import ZipFile

import torch
import cv2
import numpy as np
from tasker import Profile, Definition, Return
from tasker.storage import Storage
from tasker.tasks import Task
from tasker.tasks.torch import SimpleTrainTask
from torch import nn
from ignite import metrics, engine
from torchvision.models.segmentation import deeplabv3_resnet50


class NAIC2020RSModelTrainTask(SimpleTrainTask):
    class Model(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.add_module('deeplabv3', nn.DataParallel(deeplabv3_resnet50(pretrained=False, num_classes=8)))

        def forward(self, x: torch.Tensor):
            y: torch.Tensor = self.deeplabv3(x)
            return y['out']

    def __init__(self, prefix: Text = None):
        super(NAIC2020RSModelTrainTask, self).__init__(prefix)
        if prefix is not None:
            self.PROVIDE_STATE_KEY = f'{prefix}_train_state'
        else:
            self.PROVIDE_STATE_KEY = 'train_state'

    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        return self.Model()

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
        metrics_['miou'] = metrics.mIoU(metrics_['confusion_matrix'])


class NAIC2020RSResultDumpTask(Task):
    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        model: nn.Module = shared['model']
        timestr = datetime.now().strftime('%Y%m%d%H%M%S')
        sample_dir = Path(profile.sample_dir)
        counter = 0

        with ZipFile(Path('.') / '.tasker' / 'output' / 'naic2020rs' / f'{timestr}.zip', mode='x') as package:
            for fname in listdir(sample_dir):
                index = fname.split('.')[0]
                raw_image: np.ndarray = cv2.imread(str(sample_dir / fname), flags=cv2.IMREAD_UNCHANGED)
                raw_image = raw_image.transpose((2, 0, 1))[np.newaxis, :, :, :]
                raw_image = raw_image.astype(np.float32)
                with torch.no_grad():
                    target: torch.Tensor = model(torch.from_numpy(raw_image).to('cuda:0')).to('cpu')
                target: np.ndarray = (target.argmax(dim=1) + 1).numpy().astype(np.uint16) * 100
                target = target[0, :, :]
                with package.open(f'results/{index}.png', mode='w') as fp:
                    fp.write(cv2.imencode('.png', target)[1])
                counter += 1
                if counter % 1000 == 0:
                    logger.info(f'Dumped {counter} samples.')
            return Return.SUCCESS.value

    def require(self) -> List[Text]:
        return ['model']

    def provide(self) -> List[Text]:
        return []

    def remove(self) -> List[Text]:
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        return []


class NAIC2020RSPretrainedModelTrainTask(NAIC2020RSModelTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        model_path = profile.model_path
        with open(model_path, 'rb') as fp:
            return pickle.load(fp)
