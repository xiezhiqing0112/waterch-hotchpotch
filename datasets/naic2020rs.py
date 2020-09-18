from logging import Logger
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


class NAIC2020RS(Dataset, ProfileMixin):
    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('root_dir', str)
        ]

    @property
    def train_permutation(self):
        permutarion_path = self.root_dir / 'train' / 'permutation.npy'
        if permutarion_path.exists():
            return np.load(str(permutarion_path))
        else:
            permutation = np.random.permutation(len(listdir(self.root_dir / 'train' / 'label')))
            np.save(str(permutarion_path), permutation)
            return permutation

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
            self.index = permutation[:int(total_number * self.ratio)] + 1
        else:
            self.index = permutation[int(total_number * self.ratio):] + 1

    def __getitem__(self, item):
        return cv2.imread(
            str(self.root_dir / 'train' / 'image' / f'{item}.tif')
        ), cv2.imread(
            str(self.root_dir / 'train' / 'label' / f'{item}.png')
        ) - 1

    def __len__(self):
        return self.index.shape[0]


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
