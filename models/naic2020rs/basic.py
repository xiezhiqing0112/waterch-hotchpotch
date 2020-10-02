import pickle
from datetime import datetime
from logging import Logger, getLogger as get_logger
from multiprocessing import Pool
from os import listdir
from pathlib import Path
from typing import OrderedDict, Dict, Text, Any, List, Tuple
from zipfile import ZipFile

import torch
import cv2
import numpy as np
from numba import njit
from tasker import Profile, Definition, Return
from tasker.storage import Storage
from tasker.tasks import Task
from tasker.tasks.torch import SimpleTrainTask
from torch import nn
from sklearn.preprocessing import OneHotEncoder
from ignite import metrics, engine
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101


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


class ForceLoadModelTask(Task):
    def __init__(self, prefix: Text = None):
        if prefix is not None:
            self.MODEL_KEY = f'{prefix}_model'
        else:
            self.MODEL_KEY = 'model'

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        with open(profile.model_path, 'rb') as fp:
            shared[self.MODEL_KEY] = pickle.load(fp)

        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        return []

    def provide(self) -> List[Text]:
        return [self.MODEL_KEY]

    def remove(self) -> List[Text]:
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        return []


class NAIC2020RSResultDumpTask(Task):
    def __init__(self, prefix: Text = None):
        if prefix is not None:
            self.MODEL_KEY = f'{prefix}_model'
        else:
            self.MODEL_KEY = 'model'

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        model: nn.Module = shared[self.MODEL_KEY]
        timestr = datetime.now().strftime('%Y%m%d%H%M%S')
        sample_dir = Path(profile.sample_dir)
        counter = 0

        model.eval()

        with ZipFile(Path('.') / '.tasker' / 'output' / 'naic2020rs' / f'{timestr}.zip', mode='x') as package:
            for fname in listdir(sample_dir):
                index = fname.split('.')[0]
                raw_image: np.ndarray = cv2.imread(str(sample_dir / fname), flags=cv2.IMREAD_UNCHANGED)
                raw_image = raw_image.transpose((2, 0, 1))[np.newaxis, :, :, :]
                raw_image = raw_image.astype(np.float32) / 255.0
                with torch.no_grad():
                    target: torch.Tensor = model(torch.from_numpy(raw_image).to('cuda:0')).to('cpu')
                target: np.ndarray = (target.argmax(dim=1) + 1).numpy().astype(np.uint16) * 100
                target = target[0, :, :]
                with package.open(f'results/{index}.png', mode='w') as fp:
                    fp.write(cv2.imencode('.png', target)[1])
                counter += 1
                if counter % 1000 == 0:
                    logger.info(f'Dumped {counter} samples.')
            logger.info(f'Dumped to {package.filename}.')
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


class NAIC2020RSModel101TrainTask(NAIC2020RSModelTrainTask):
    class Model(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.add_module('deeplabv3', nn.DataParallel(deeplabv3_resnet101(pretrained=False, num_classes=8)))

        def forward(self, x: torch.Tensor):
            y: torch.Tensor = self.deeplabv3(x)
            return y['out']


class NAIC2020RSResultFusionTask(Task):
    class FusionWorker:
        def __init__(self):
            self._counter = 0
            self.logger = get_logger()
            self.onehot_encoder = OneHotEncoder(categories=8, dtype=np.float32)

        @property
        def counter(self):
            self._counter += 1
            return self._counter

        def _fusion(self, label: np.ndarray) -> np.ndarray:
            onehot: np.ndarray = np.concatenate(tuple(map(
                lambda index: (label == index)[:, :, :, np.newaxis],
                range(8)
            )), axis=3).sum(axis=2)

            return onehot.argmax(axis=2).astype(np.uint16)

        def __call__(self, params):
            rname, concatenated = params
            min_label: np.ndarray = concatenated // 100 - 1
            fusion_label = self._fusion(min_label)
            target = (fusion_label + 1) * 100

            counter = self.counter
            if counter % 100 == 0:
                self.logger.info(f'Generated {counter} samples of models fusion.')
            else:
                self.logger.debug(f'Generated {counter} samples of models fusion.')
            return rname, target

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        assert 'result_zips' in profile
        assert 'sample_dir' in profile
        assert 'num_workers' in profile
        timestr = datetime.now().strftime('%Y%m%d%H%M%S')
        num_workers = profile.num_workers

        zip_files: Tuple[ZipFile] = tuple(map(
            lambda it: ZipFile(it, mode='r'),
            profile.result_zips
        ))
        target_zip = ZipFile(Path('.') / '.tasker' / 'output' / 'naic2020rs' / f'{timestr}F.zip', mode='x')

        worker = self.FusionWorker()
        if num_workers > 0:
            pool = Pool(num_workers)
            pool_map = pool.map
        else:
            pool_map = map

        def _load_png(zip_file: ZipFile, rname: Text) -> np.ndarray:
            with zip_file.open(f'results/{rname}', mode='r') as fp:
                array = cv2.imdecode(
                    np.asarray(bytearray(fp.read())),
                    flags=cv2.IMREAD_UNCHANGED
                )
                return array[:, :, np.newaxis]

        for rname, target in pool_map(worker, map(
                lambda rname: (rname, np.concatenate(tuple(map(
                    lambda it: _load_png(it, rname),
                    zip_files
                )), axis=2)),
                map(
                    lambda it: it.replace('.tif', '.png'),
                    listdir(profile.sample_dir)
                ))):
            with target_zip.open(f'results/{rname}', mode='w') as fp:
                fp.write(cv2.imencode('.png', target)[1])

        logger.info(f'Dumped to {target_zip.filename}.')

        for zip_file in zip_files:
            zip_file.close()
        target_zip.close()

        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        return []

    def provide(self) -> List[Text]:
        return []

    def remove(self) -> List[Text]:
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        return []


class NAIC2020RSModelFusionTask(Task):
    class FusionWorker:
        def __call__(self, *args, **kwargs):
            pass

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        assert 'sample_dir' in profile
        assert 'model_paths' in profile

        timestr = datetime.now().strftime('%Y%m%d%H%M%S')
        sample_dir = Path(profile.sample_dir)
        counter = 0

        def _load_model(path: Text):
            with open(path, 'rb') as fp:
                model = pickle.load(fp)
                assert isinstance(model, nn.Module)
                return model

        models = tuple(map(_load_model, profile.model_paths))
        for model in models:
            model.eval()

        with ZipFile(Path('.') / '.tasker' / 'output' / 'naic2020rs' / f'{timestr}M.zip', mode='x') as target_zip:
            for filename in listdir(sample_dir):
                raw_image: np.ndarray = cv2.imread(str(sample_dir / filename), flags=cv2.IMREAD_UNCHANGED)
                raw_image = raw_image.transpose((2, 0, 1))[np.newaxis, :, :, :]
                raw_image = raw_image.astype(np.float32) / 255.0
                index = filename.split('.')[0]
                with torch.no_grad():
                    target = np.concatenate(tuple(map(
                        lambda it: it.numpy().transpose((2, 3, 0, 1)),
                        map(
                            lambda model: model(torch.from_numpy(raw_image).to('cuda:0')).softmax(dim=1).to('cpu'),
                            models
                        )
                    )), axis=2)
                    fusion: np.ndarray = target.mean(axis=2).argmax(axis=2)
                    result = ((fusion + 1) * 100).astype(np.uint16)
                    with target_zip.open(f'results/{index}.png', mode='w') as fp:
                        fp.write(cv2.imencode('.png', result)[1])

                    counter += 1
                    if counter % 1000 == 0:
                        logger.info(f'Dumped {counter} samples by model fusion.')

        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        return []

    def provide(self) -> List[Text]:
        return []

    def remove(self) -> List[Text]:
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        return []
