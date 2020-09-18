from logging import getLogger
from multiprocessing.dummy import Pool
from os import listdir, path, makedirs
from typing import List, Text

import numpy as np
import torch
from torch.utils.data import Dataset
from tasker import Definition
from tasker.mixin import ProfileMixin, value
from tasker.tasks.torch import SimpleDataLoaderTask
from sklearn import preprocessing


class MHealth(Dataset, ProfileMixin):
    logger = getLogger('datasets.mhealth.MHealth')

    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('root_dir', str),
            value('num_workers', int),
            value('window_size', int),
            value('window_stride', int),
            value('filter_type', str),
            value('filter_by', list, [str]),
            value('preprocess', list, [str]),
        ]

    @classmethod
    def _cache_load(cls, raw_dir: Text, processed_dir: Text, fname: Text) -> np.ndarray:
        cls.logger.debug(f'Loading {fname}')
        cache_path = path.join(processed_dir, fname.replace('.log', '.raw.npy'))
        raw_path = path.join(raw_dir, fname)
        if path.exists(cache_path) and path.getsize(cache_path) > 0:
            return np.load(cache_path)
        else:
            raw_array = np.loadtxt(raw_path, dtype=np.float32)
            np.save(cache_path, raw_array)
            return raw_array

    @classmethod
    def _apply_window(cls, array: np.ndarray, window_size: int, window_stride: int) -> np.ndarray:
        rows, columns = array.shape
        return np.concatenate(tuple(filter(
            lambda it: np.argmax(np.bincount(it[0, -1, :].astype(np.long))) != 0,
            map(
                lambda it: array[np.newaxis, it * window_stride: it * window_stride + window_size, :].transpose(
                    (0, 2, 1)),
                filter(
                    lambda it: (it * window_stride + window_size) < rows,
                    range(rows // window_stride)
                )
            )
        )), axis=0)

    @classmethod
    def _preprocess_robust(cls, array: np.ndarray) -> np.ndarray:
        frames, channels, window_size = array.shape
        scaled_array = array.transpose((1, 0, 2)).reshape((channels, -1))
        scaled_array = preprocessing.robust_scale(scaled_array, axis=1)
        return preprocessing.robust_scale(scaled_array, axis=1).reshape(channels, frames, window_size).transpose(1, 0, 2)

    @classmethod
    def _preprocess_minmax(cls, array: np.ndarray) -> np.ndarray:
        frames, channels, window_size = array.shape
        scaled_array = array.transpose((1, 0, 2)).reshape((channels, -1))
        scaled_array = preprocessing.robust_scale(scaled_array, axis=1)
        return preprocessing.minmax_scale(scaled_array, axis=1).reshape(channels, frames, window_size).transpose(1, 0, 2)

    def __init__(self, **kwargs):
        root_dir = kwargs['root_dir']
        num_workers = kwargs['num_workers']

        if num_workers > 0:
            pool = Pool(num_workers)
            pool_map = pool.map
        else:
            pool_map = map

        window_size = kwargs['window_size']
        window_stride = kwargs['window_stride']
        filter_type = kwargs['filter_type']
        assert filter_type in ('if', 'if_not')
        filter_by = kwargs['filter_by']
        for it in filter_by:
            assert it in tuple(pool_map(lambda idx: f'subject{idx}', range(1, 11)))
        preprocess = kwargs['preprocess']

        includes = tuple(filter(
            lambda it: it in filter_by if filter_type == 'if' else it not in filter_by,
            pool_map(
                lambda it: f'subject{it}',
                range(1, 11)
            )
        ))

        self.logger.info(f'Includes subjects: {", ".join(includes)}')

        raw_dir = path.join(root_dir, 'raw')
        processed_dir = path.join(root_dir, 'processed')
        if not path.exists(processed_dir):
            makedirs(processed_dir)

        raw_subjects = dict(pool_map(
            lambda it: (it[8:-4], self._cache_load(raw_dir, processed_dir, it)),
            filter(
                lambda it: it.startswith('mHealth_') and it.endswith('.log'),
                listdir(raw_dir)
            )
        ))

        windowed_subjects = dict(pool_map(
            lambda it: (it[0], self._apply_window(it[1], window_size, window_stride)),
            raw_subjects.items()
        ))

        del raw_subjects

        raw = np.concatenate(tuple(pool_map(
            lambda it: it[1],
            filter(
                lambda it: it[0] in includes,
                windowed_subjects.items()
            )
        )), 0)

        del windowed_subjects
        self.logger.debug(f'Raw array with shape {raw.shape}')

        self.features = raw[:, :-1, :]
        for name in preprocess:
            method = getattr(self, f'_preprocess_{name}', lambda x: x)
            self.features = method(self.features)

        self.labels = np.array(tuple(pool_map(
            lambda it: np.argmax(np.bincount(it)),
            raw[:, -1, :].astype(np.long)
        ))) - 1

    def __getitem__(self, item):
        return torch.tensor(self.features[item]), torch.tensor(self.labels[item])

    def __len__(self):
        return self.labels.shape[0]


class MHealthDataLoaderTask(SimpleDataLoaderTask):
    pass
