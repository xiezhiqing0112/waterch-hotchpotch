from collections import OrderedDict
from logging import Logger, getLogger as get_logger
from typing import List, Text
from pathlib import Path
from os import listdir

import cv2
import numpy as np
from tasker import Definition, Profile
from tasker.mixin import ProfileMixin, value
from tasker.storage import Storage
from tasker.tasks.torch import SimpleDataLoaderTask
from torch.utils.data import Dataset, BatchSampler, RandomSampler


def image_minmax(img):
    return (np.clip(img, 0, 255) / 255.0).astype(np.float32)


def label_type(label):
    return label.astype(np.int64)


def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def normal_random_noise(img, op=lambda s, n: s + n):
    return op(img, np.random.normal(size=img.shape)).astype(img.dtype)


class NAIC2020RS(Dataset, ProfileMixin):
    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('root_dir', str)
        ]

    @property
    def train_permutation(self):
        permutation_path = self.root_dir / 'train' / 'permutation.npy'
        if permutation_path.exists():
            return np.load(str(permutation_path))
        else:
            permutation = np.random.permutation(len(listdir(self.root_dir / 'train' / 'label')))
            np.save(str(permutation_path), permutation)
            return permutation

    @property
    def train_filelist(self):
        filelist_path = self.root_dir / 'train' / 'filelist.npy'
        if filelist_path.exists():
            return np.load(str(filelist_path))
        else:
            filelist = np.array(tuple(map(
                lambda it: it.split('.')[0],
                listdir(self.root_dir / 'train' / 'label')
            )))
            np.save(str(filelist_path), filelist)
            return filelist

    def __init__(self, **kwargs):
        assert 'root_dir' in kwargs
        assert 'train' in kwargs
        assert 'ratio' in kwargs

        self.root_dir = Path(kwargs['root_dir'])
        self.train: bool = kwargs['train']
        permutation = self.train_permutation
        self.ratio = kwargs['ratio']
        total_number = permutation.shape[0]

        if self.train:
            self.index = permutation[:int(total_number * self.ratio)]
        else:
            self.index = permutation[int(total_number * self.ratio):]

    process_methods = OrderedDict((
        (
            0, (  # Raw
                lambda feature: feature,
                lambda label: label
            )
        ),
        (
            1, (  # Flip axis 0
                lambda feature: np.flip(feature, axis=1),
                lambda label: np.flip(label, axis=0),
            )
        ),
        (
            2, (  # Flip axis 1
                lambda feature: np.flip(feature, axis=2),
                lambda label: np.flip(label, axis=1),
            )
        ),
        (
            3, (  # Rotate 90
                lambda feature: np.rot90(feature, 1, (1, 2)),
                lambda label: np.rot90(label, 1, (0, 1))
            )
        ),
        (
            4, (  # Rotate 180
                lambda feature: np.rot90(feature, 2, (1, 2)),
                lambda label: np.rot90(label, 2, (0, 1)),
            )
        ),
        (
            5, (  # Rotate 270
                lambda feature: np.rot90(feature, 3, (1, 2)),
                lambda label: np.rot90(label, 3, (0, 1)),
            )
        ),
        (
            6, (  # Blur 2x2
                lambda feature: cv2.blur(feature.transpose((1, 2, 0)), (2, 2)).transpose((2, 0, 1)),
                lambda label: label,
            )
        ),
        (
            7, (  # Blur 3x3
                lambda feature: cv2.blur(feature.transpose((1, 2, 0)), (3, 3)).transpose((2, 0, 1)),
                lambda label: label,
            )
        ),
        (
            8, (  # Gamma 2.0
                lambda feature: random_gamma_transform(feature.transpose((1, 2, 0)), 2.0).transpose((2, 0, 1)),
                lambda label: label,
            )
        ),
        (
            9, (  # Gamma 0.5
                lambda feature: random_gamma_transform(feature.transpose((1, 2, 0)), 0.5).transpose((2, 0, 1)),
                lambda label: label,
            )
        ),
        (
            10, (  # Noise plus
                lambda feature: normal_random_noise(feature.transpose((1, 2, 0)), lambda s, n: s + n).transpose(
                    (2, 0, 1)),
                lambda label: label,
            )
        ),
        (
            11, (  # Noise minus
                lambda feature: normal_random_noise(feature.transpose((1, 2, 0)), lambda s, n: s - n).transpose(
                    (2, 0, 1)),
                lambda label: label,
            )
        ),
        (
            12, (  # Transpose
                lambda feature: feature.transpose((0, 2, 1)),
                lambda label: label.transpose((1, 0)),
            )
        )
    ))

    def __getitem__(self, item):
        logger = get_logger('datasets.naic2020rs.NAIC2020RS.__getitem__')
        # logger.info(item)

        process_index = int(item // self.index.shape[0])
        feature_process, label_process = self.process_methods[process_index]
        raw_item = item % self.index.shape[0]

        feature_path = self.root_dir / 'train' / 'image' / f'{self.train_filelist[self.index[raw_item]]}.tif'
        label_path = self.root_dir / 'train' / 'label' / f'{self.train_filelist[self.index[raw_item]]}.png'
        # logger.info(feature_path)
        # logger.info(label_path)

        feature: np.ndarray = cv2.imread(
            str(feature_path),
            flags=cv2.IMREAD_UNCHANGED
        )  # Shape (256, 256, 3)
        label: np.ndarray = cv2.imread(
            str(label_path),
            flags=cv2.IMREAD_UNCHANGED
        )  # Shape (256, 256, 3)

        feature = feature.transpose((2, 0, 1))  # Shape (3, 256, 256)
        label = label // 100 - 1  # Shape (256, 256)

        return image_minmax(feature_process(feature)), label_type(label_process(label))

    def __len__(self):
        return self.index.shape[0] * len(self.process_methods)


class NAIC2020RSDataLoaderTask(SimpleDataLoaderTask):
    def create_sampler(
            self, dataset: Dataset, batch_sampler: bool, profile: Profile, shared: Storage, logger: Logger
    ):
        assert batch_sampler

        return BatchSampler(
            RandomSampler(dataset),
            batch_size=profile.batch_size,
            drop_last=profile.drop_last
        )
