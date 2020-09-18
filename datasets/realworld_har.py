from collections import OrderedDict
from functools import reduce
from itertools import product, chain
from typing import List
from os import listdir, path
from zipfile import ZipFile

import pandas as pd
import numpy as np
from tasker import Definition
from tasker.mixin import ProfileMixin, value
from torch.utils.data import Dataset


class RealWorldHAR(Dataset, ProfileMixin):
    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('root_dir', str),
            value('frame_period', int),
            value('filter_type', list, [str]),
            value('filter_by', str)
        ]

    def __init__(self, **kwargs):
        assert 'root_dir' in kwargs
        assert 'frame_period' in kwargs
        assert 'filter_type' in kwargs
        assert 'filter_by' in kwargs

        self.sample_bound = 45
        self.root_dir = kwargs['root_dir']
        self.frame_period = kwargs['frame_period']
        self.filter_type = kwargs['filter_type']
        self.filter_by = kwargs['filter_by']

        raw_data = OrderedDict(map(lambda proband: (proband, self._load_proband(proband)), listdir(self.root_dir)))
        pass

    def _load_proband(self, proband_name):
        labels = (
            'standing', 'sitting', 'lying', 'walking',
            'running', 'jumping', 'climbingup', 'climbingdown'
        )
        sensors = ('acc', 'gyr', 'mag')
        positions = (
            'chest', 'forearm', 'head', 'shin',
            'thigh', 'upperarm', 'waist'
        )

        def _sample_frame(frame):
            permutation = np.sort(np.random.permutation(frame[1].shape[0]))[:self.sample_bound * self.frame_period]
            return frame[0], frame[1].iloc[permutation, :]

        def _load_zipfile_csv(zip_file, filename):
            with zip_file.open(filename) as fp:
                frame = pd.read_csv(fp)
                frame['time_index'] = frame['attr_time'] // (1000 * self.frame_period)
                return OrderedDict(map(
                    lambda it: (it[0], it[1].loc[:, ('attr_x', 'attr_y', 'attr_y')]),
                    map(
                        _sample_frame,
                        filter(
                            lambda it: it[1].shape[0] > self.sample_bound * self.frame_period,
                            frame.groupby('time_index')
                        )
                    )
                ))

        def _load_zipfile(filename):
            try:
                with ZipFile(path.join(self.root_dir, proband_name, 'data', filename)) as zip_file:
                    raw_dict = OrderedDict(map(
                        lambda it: (it[0], _load_zipfile_csv(zip_file, it[1])),
                        map(
                            lambda position: (
                                position,
                                tuple(filter(lambda it: position in it.filename, zip_file.filelist))[0]
                            ),
                            positions
                        )
                    ))
                    common_indexes = tuple(sorted(reduce(
                        lambda s1, s2: s1 & s2,
                        map(lambda it: set(it.keys()), raw_dict.values())
                    )))
                    return OrderedDict(map(
                        lambda position: (position, OrderedDict(map(
                            lambda idx: (idx, raw_dict[position][idx].to_numpy().astype(np.float32).transpose()[np.newaxis, :, :]),
                            common_indexes
                        ))),
                        sorted(raw_dict)
                    ))
            except IndexError:
                return OrderedDict()

        per_sensor_label = OrderedDict(map(
            lambda it: (it[0], _load_zipfile(it[1])),
            map(
                lambda it: (it, f'{it[0]}_{it[1]}_csv.zip'),
                product(sensors, labels)
            )
        ))

        per_sensor_indexes = tuple(zip(map(
            lambda sensor: tuple(map(
                lambda it: reduce(
                    lambda s1, s2: s1 & s2,
                    map(
                        lambda it1: set(it1.keys()),
                        it[1].values()
                    )
                ) if it[1] else set(),
                filter(
                    lambda it: it[0][0] == sensor,
                    per_sensor_label.items()
                )
            )),
            sensors
        )))

        combined_indexes = reduce(
            lambda s1, s2: s1 & s2,
            map(
                lambda it: reduce(
                    lambda s1, s2: s1 | s2,
                    it[0]
                ),
                per_sensor_indexes
            )
        )

        features = np.concatenate(tuple(map(
            lambda sensor: np.concatenate(tuple(map(  # Single sensor group.
                lambda it_z: np.concatenate(tuple(map(  # Single label file.
                    lambda it_p: np.concatenate(tuple(map(  # Data frames for each position.
                        lambda it_t: it_t[1],  # Frames per time index.
                        sorted(filter(
                            lambda it_t: it_t[0] in combined_indexes,
                            it_p.items()
                        ), key=lambda it_t: it_t[0])
                    )), axis=0),
                    it_z.values()
                )), axis=1),
                map(lambda key: per_sensor_label[key], filter(lambda it: it[0] == sensor, per_sensor_label))
            )), axis=0),
            sensors
        )), axis=1)

        labels = np.concatenate(tuple(map(
            lambda sensor: sensor,
            sensors
        )))

        return features