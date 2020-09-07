import pickle
from collections import deque
from enum import Enum
from functools import reduce
from logging import getLogger, Logger
from multiprocessing.dummy import Pool
from os import path
from typing import List
from typing import Text

import numpy as np
import torch
from gc import collect
from sklearn.decomposition import PCA
from torch.utils import data
from torch.utils.data import Dataset, BatchSampler, RandomSampler
from waterch.tasker import Definition, value, Profile
from waterch.tasker.mixin import ProfileMixin
from waterch.tasker.storage import Storage
from waterch.tasker.tasks.torch import SimpleDataLoaderTask


def QV_cross_1d(v0, v1, v2, Q0, Q1, Q2, Q3):
    q0 = -Q1 * v0 - Q2 * v1 - Q3 * v2
    q1 = Q0 * v0 + Q2 * v2 - Q3 * v1
    q2 = Q0 * v1 + Q3 * v0 - Q1 * v2
    q3 = Q0 * v2 + Q1 * v1 - Q2 * v0

    i = -q0 * Q1 + q1 * Q0 - q2 * Q3 + q3 * Q2
    j = -q0 * Q2 + q2 * Q0 - q3 * Q1 + q1 * Q3
    k = -q0 * Q3 + q3 * Q0 - q1 * Q2 + q2 * Q1

    return i, j, k


class HuaweiSussex(Dataset, ProfileMixin):
    logger = getLogger('datasets.shl.HuaweiSussex')

    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('root_dir', str),
            value('num_workers', int),
            value('process', str),
            value('subsets', list, [str]),
            value('label_mapping', str),
            value('remove_sensor', list, [str])
        ]

    def __init__(self, *args, **kwargs):
        # Fill all attributes in kwargs into this object.
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Check integrity of parameters.
        assert hasattr(self, 'root_dir')
        assert hasattr(self, 'num_workers')
        assert hasattr(self, 'process')
        assert hasattr(self, 'subsets')
        assert hasattr(self, 'label_mapping')
        assert hasattr(self, 'remove_sensor')
        # Build thread pool
        self.pool = Pool(self.num_workers)
        # Initialize dataset
        self._features_dict = self._load_dataset()
        self.logger.info('Features loaded.')
        self._label = self._load_label()
        self.logger.info('Label loaded.')
        self._convert()
        self.logger.info('Dataset converted.')
        self._clean()
        # Print idx_map keys
        self.logger.info(' '.join(self.idx_map.keys()))
        self.logger.info('Done.')

    class Field(Enum):
        label = ('Label', ())
        ACCELEROMETER = ('Acc', ('x', 'y', 'z'))
        GRAVITY = ('Gra', ('x', 'y', 'z'))
        GYROSCOPE = ('Gyr', ('x', 'y', 'z'))
        LINEAR_ACCELERATION = ('LAcc', ('x', 'y', 'z'))
        MAGNETIC_FIELD = ('Mag', ('x', 'y', 'z'))
        ORIENTATION = ('Ori', ('w', 'x', 'y', 'z'))
        PRESSURE = ('Pressure', ())

    def _load_single_source(self, filename: Text):
        """
        Parse single .txt file by file path.
        :param filename: file path of single source file
        :return: ndarray of data
        """
        self.logger.debug(filename)
        pickle_filename = f'{filename}.pkl'
        if path.exists(pickle_filename):
            with open(pickle_filename, 'rb') as fp:
                try:
                    return pickle.load(fp)
                except Exception:
                    values = np.loadtxt(fname=filename, dtype=np.float32)
                    with open(pickle_filename, 'wb') as fp:
                        pickle.dump(values, fp)
                    return values
        else:
            values = np.loadtxt(fname=filename, dtype=np.float32)
            with open(pickle_filename, 'wb') as fp:
                pickle.dump(values, fp)
            return values

    def _load_raw_sensor(self, folder, sensor) -> torch.Tensor:
        """
        Parse sensor array of single subset.
        :param folder:
        :param sensor:
        :return:
        """
        name, axes = sensor.value
        if len(axes) > 0:
            raw_value = np.concatenate(tuple(map(
                lambda ax: ax.reshape(ax.shape[0], 1, ax.shape[1]),
                map(
                    lambda ax: self._load_single_source(path.join(folder, f'{name}_{ax}.txt')),
                    axes
                )
            )), axis=1)
        else:
            raw_value = self._load_single_source(path.join(folder, f'{name}.txt'))
            raw_value = raw_value.reshape(raw_value.shape[0], 1, raw_value.shape[1])
        return raw_value

    def _load_raw(self):
        raw = tuple(self.pool.map(
            lambda subset: dict(map(
                lambda sensor: (
                    sensor.name,
                    self._load_raw_sensor(subset, sensor)
                ),
                filter(
                    lambda
                        sensor: sensor != self.Field.label and sensor.name not in self.remove_sensor or sensor == self.Field.ORIENTATION,
                    self.Field
                )
            )),
            map(
                lambda subset: path.join(self.root_dir, subset),
                self.subsets
            )
        ))
        return raw

    @property
    def idx_map(self):
        fields = list(filter(
            lambda sensor: sensor.name not in self.remove_sensor,
            self.Field
        ))
        return dict(map(
            lambda index: (fields[index].name, index),
            range(len(fields))
        ))

    def _process_pca(self, raw_dataset, params):
        components = eval(params[0])

        def pca_extract(array: np.ndarray):
            pca = PCA(components)
            trans = pca.fit_transform(array.transpose()).transpose()
            del pca
            trans[np.isnan(trans)] = 0.0
            return trans

        return dict(map(
            lambda item: (item[0], np.concatenate(tuple(map(
                lambda array: array.reshape(1, *array.shape),
                map(
                    pca_extract,
                    map(
                        lambda index: item[1][index, :, :],
                        range(item[1].shape[0])
                    )
                )
            )), axis=0)) if item[1].shape[1] > 1 else item,
            raw_dataset.items()
        ))

    def _process_quaternion(self, raw_dataset, params):
        orientation = raw_dataset[self.Field.ORIENTATION.name]
        frames, _, framesize = orientation.shape

        def _quaternion_extract(array: np.ndarray):
            if array.shape[1] != 3:
                return array
            else:
                x, y, z = array[:, 0, :].reshape(-1), array[:, 1, :].reshape(-1), array[:, 2, :].reshape(-1)
                o_w, o_x, o_y, o_z = \
                    orientation[:, 0, :].reshape(-1), \
                    orientation[:, 1, :].reshape(-1), \
                    orientation[:, 2, :].reshape(-1), \
                    orientation[:, 3, :].reshape(-1)
                i, j, k = QV_cross_1d(x, y, z, o_w, o_x, o_y, o_z)
                return np.concatenate(tuple(map(
                    lambda axis: axis.reshape(frames, 1, framesize),
                    (i, j, k, np.sqrt(i ** 2 + j ** 2 + k ** 2))
                )), axis=1)

        return dict(map(
            lambda item: (item[0], _quaternion_extract(item[1])),
            filter(
                lambda item: item[0] != self.Field.ORIENTATION.name,
                raw_dataset.items()
            )
        ))

    def _process_quaternion_components(self, raw_dataset, params):
        quaternion_dataset = self._process_quaternion(raw_dataset, params)

        def _components_extract(array):
            if array.shape[1] != 4:
                return array
            else:
                frames, _, frame_size = array.shape
                x, y, z, f = array[:, 0, :].reshape(-1), \
                             array[:, 1, :].reshape(-1), \
                             array[:, 2, :].reshape(-1), \
                             array[:, 3, :].reshape(-1)
                h = np.sqrt(x ** 2 + y ** 2)
                i = np.arcsin(z / f)
                if np.isnan(i).sum() != 0:
                    i[np.isnan(i)] = 0.0
                d = np.arcsin(y / h)
                if np.isnan(d).sum() != 0:
                    d[np.isnan(d)] = 0.0
                return np.concatenate(tuple(map(
                    lambda axis: axis.reshape(frames, 1, frame_size),
                    (x, y, z, h, f, i, d)
                )), axis=1)

        return dict(map(
            lambda item: (item[0], _components_extract(item[1])),
            quaternion_dataset.items()
        ))

    def _process_quaternion_pca(self, raw_dataset, params):
        quaternion = self._process_quaternion(raw_dataset, params)
        return self._process_pca(quaternion, params)

    def _process_raw(self, raw_dataset, params):
        return raw_dataset

    def _concat(self, raw_dataset):
        return dict(map(
            lambda key: (key, np.concatenate(tuple(map(
                lambda subset: subset[key],
                raw_dataset
            )), axis=0)),
            raw_dataset[0].keys()
        ))

    def _process(self, raw_dataset):
        split = self.process.split(',')
        method = split[0].replace('-', '_')
        process_fun = getattr(self, f'_process_{method}', self._process_raw)
        self.logger.debug(process_fun)
        processed = process_fun(raw_dataset, split[1:])
        collect()
        return processed

    def _normalize(self, raw_dataset):
        def normalize_axis(array):
            percentile = np.percentile(array, (25, 75), interpolation='midpoint')
            percentile_range = percentile[1] - percentile[0]
            lower_limit = percentile[0] - 1.5 * percentile_range
            upper_limit = percentile[1] + 1.5 * percentile_range
            clip = np.clip(array, lower_limit, upper_limit)
            return (clip - lower_limit) / (upper_limit - lower_limit)

        return dict(self.pool.map(
            lambda item: (item[0], np.concatenate(tuple(map(
                lambda array: array.reshape(array.shape[0], 1, array.shape[1]),
                map(
                    lambda axis: normalize_axis(item[1][:, axis, :]),
                    range(item[1].shape[1])
                )
            )), axis=1)),
            raw_dataset.items()
        ))

    def _load_dataset(self):
        raw_dataset = deque(self._load_raw())
        result = []
        while len(raw_dataset) > 0:
            dataset = raw_dataset.pop()
            result.append(self._normalize(self._process(dataset)))
        return self._concat(result)

    def _load_single_label(self, filename):
        self.logger.debug(filename)
        pickle_filename = f'{filename}.pkl'
        if path.exists(pickle_filename):
            with open(pickle_filename, 'rb') as fp:
                try:
                    return pickle.load(fp)
                except Exception:
                    values = np.loadtxt(fname=filename, dtype=np.long)
                    with open(pickle_filename, 'wb') as fp:
                        pickle.dump(values, fp)
                    return values
        else:
            values = np.loadtxt(fname=filename, dtype=np.long)
            with open(pickle_filename, 'wb') as fp:
                pickle.dump(values, fp)
            return values

    def _load_raw_label(self):
        return np.concatenate(tuple(map(
            lambda array: array.reshape(array.shape[0], 1, array.shape[1]),
            map(
                lambda filename: self._load_single_label(filename),
                map(
                    lambda subset: path.join(self.root_dir, subset, 'Label.txt'),
                    self.subsets
                )
            )
        )), axis=0)

    def _label_mapping_raw(self, label):
        return label - 1

    def _label_mapping_ocbr(self, label):
        # Definition by WaterCH
        # 1 - Still, 2 - Walk, 3 - Run, 4 - Bike, 5 - Car, 6 - Bus, 7 - Train, 8 - Subway
        # Mapping {Still, Walk, Run, Bike} -> Other, Car -> Car, Bus -> Bus, {Train, Subway} -> Rail
        label[label == 1] = 0
        label[label == 2] = 0
        label[label == 3] = 0
        label[label == 4] = 0
        label[label == 5] = 1
        label[label == 6] = 2
        label[label == 7] = 3
        label[label == 8] = 3
        return label

    def _label_mapping_ts(self, label):
        # Definition by WaterCH
        # 1 - Still, 2 - Walk, 3 - Run, 4 - Bike, 5 - Car, 6 - Bus, 7 - Train, 8 - Subway
        # Mapping Train -> Train, Subway -> Subway, other classes will be removed.
        label[label == 1] = -1
        label[label == 2] = -1
        label[label == 3] = -1
        label[label == 4] = -1
        label[label == 5] = -1
        label[label == 6] = -1
        label[label == 7] = 0
        label[label == 8] = 1
        return label

    def _label_mapping_ocbts(self, label):
        # Definition by WaterCH
        # 1 - Still, 2 - Walk, 3 - Run, 4 - Bike, 5 - Car, 6 - Bus, 7 - Train, 8 - Subway
        # Mapping {Still, Walk, Run, Bike} -> Other, Car -> Car, Bus -> Bus, Train -> Train, Subway -> Subway
        label[label == 1] = 0
        label[label == 2] = 0
        label[label == 3] = 0
        label[label == 4] = 0
        label[label == 5] = 1
        label[label == 6] = 2
        label[label == 7] = 3
        label[label == 8] = 4
        return label

    def _label_mapping_owcbr(self, label):
        # Modified definition of 5 classes.
        # 1 - Still, 2 - Walk, 3 - Run, 4 - Bike, 5 - Car, 6 - Bus, 7 - Train, 8 - Subway
        # Mapping {Still, Run, Bike} -> Other, Walk -> Walk, Car -> Car, Bus -> Bus, {Train, Subway} -> Rail
        label[label == 1] = 0
        label[label == 2] = 1
        label[label == 3] = 0
        label[label == 4] = 0
        label[label == 5] = 2
        label[label == 6] = 3
        label[label == 7] = 4
        label[label == 8] = 4
        return label

    def _label_mapping_owcbts(self, label):
        # Modified definition of 6 classes.
        # 1 - Still, 2 - Walk, 3 - Run, 4 - Bike, 5 - Car, 6 - Bus, 7 - Train, 8 - Subway
        # Mapping {Still, Run, Bike} -> Other, Walk -> Walk, Car -> Car, Bus -> Bus, Train -> Train, Subway -> Subway
        label[label == 1] = 0
        label[label == 2] = 1
        label[label == 3] = 0
        label[label == 4] = 0
        label[label == 5] = 2
        label[label == 6] = 3
        label[label == 7] = 4
        label[label == 8] = 5
        return label

    def _label_mapping_wcbr(self, label):
        # Modified definition of 4 classes.
        # 1 - Still, 2 - Walk, 3 - Run, 4 - Bike, 5 - Car, 6 - Bus, 7 - Train, 8 - Subway
        # Mapping Walk -> Walk, Car -> Car, Bus -> Bus, {Train, Subway} -> Rail, {Still, Run, Bike} will be removed
        label[label == 1] = -1
        label[label == 2] = 0
        label[label == 3] = -1
        label[label == 4] = -1
        label[label == 5] = 1
        label[label == 6] = 2
        label[label == 7] = 3
        label[label == 8] = 3
        return label

    def _label_mapping_wcbts(self, label):
        # Modified definition of 5 classes.
        # 1 - Still, 2 - Walk, 3 - Run, 4 - Bike, 5 - Car, 6 - Bus, 7 - Train, 8 - Subway
        # Mapping Walk -> Walk, Car -> Car, Bus -> Bus, Train -> Train, Subway -> Subway
        # {Still, Run, Bike} will be removed
        label[label == 1] = -1
        label[label == 2] = 0
        label[label == 3] = -1
        label[label == 4] = -1
        label[label == 5] = 1
        label[label == 6] = 2
        label[label == 7] = 3
        label[label == 8] = 4
        return label

    def _label_mapping(self, label):
        label_mapping_function = getattr(self, f'_label_mapping_{self.label_mapping}', self._label_mapping_raw)
        return label_mapping_function(label)

    def _load_label(self):
        return self._label_mapping(self._load_raw_label())

    def _convert(self):
        if self._features_dict is not None and self._label is not None:
            self._features_dict = dict(map(
                lambda item: (self.idx_map[item[0]], torch.from_numpy(item[1])),
                filter(
                    lambda item: item[0] in self.idx_map,
                    self._features_dict.items()
                )
            ))
            self._label = torch.from_numpy(self._label)
        else:
            raise RuntimeError('Please load features and label.')

    def _clean(self):
        not_nan_mask = reduce(
            lambda t1, t2: torch.logical_and(t1, t2),
            map(
                lambda t: (~torch.isnan(t)).sum(dim=(1, 2)) == t.shape[1] * t.shape[2],
                self._features_dict.values()
            )
        )
        valid_index = list(filter(
            lambda index: True,
            filter(
                lambda index: self._label[index, 0, 0] >= 0 and not_nan_mask[index],
                range(self._label.shape[0])
            )
        ))
        self._label = self._label[valid_index]
        self._features_dict = dict(map(
            lambda item: (item[0], torch.cat(list(map(
                lambda tensor: tensor.reshape(1, *tensor.size()),
                map(
                    lambda index: item[1][index],
                    valid_index
                )
            )))),
            self._features_dict.items()
        ))

    def __getitem__(self, item):
        if isinstance(item, int):
            slice = {0: self._label[item]}
            slice.update(map(
                lambda element: (element[1], self._features_dict[element[1]][item]),
                filter(
                    lambda element: element[0] != self.Field.label.name,
                    self.idx_map.items()
                )
            ))
            return slice
        elif isinstance(item, str):
            if item == self.Field.label.name:
                return self._label
            else:
                return self._features_dict[self.idx_map[item]]
        else:
            raise KeyError(item)

    def __len__(self):
        return self._label.shape[0]


class HuaweiSussexDataLoaderTask(SimpleDataLoaderTask):
    def create_sampler(
            self, dataset: data.Dataset, batch_sampler: bool, profile: Profile, shared: Storage, logger: Logger
    ):
        assert batch_sampler

        return BatchSampler(
            RandomSampler(dataset),
            batch_size=profile.batch_size,
            drop_last=profile.drop_last
        )
