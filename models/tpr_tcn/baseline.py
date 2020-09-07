from logging import Logger
from typing import List, Text

import numpy as np
import torch
from lightgbm.sklearn import LGBMClassifier
from scipy.signal import butter, sosfilt
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import robust_scale
from sklearn.svm import SVC
from torch import nn
from waterch.tasker import Definition, Profile, Return
from waterch.tasker.storage import Storage
from waterch.tasker.tasks import Task
from xgboost.sklearn import XGBClassifier

from datasets.htc_tmd import HTCTMD
from models.densenet import DenseNetLinear
from models.embracenet import EmbraceNet
from models.tpr_tcn.tcn import TemporalConvNetworkTrainTask


class BaselineTask(Task):
    def baseline_xiong(self, profile: Profile, shared: Storage, logger: Logger, converted):
        a_std = converted[1].std(-1)
        g_mean = converted[3].mean(-1)
        g_std = converted[3].std(-1)
        m_over_0_count = (converted[2] >= 0.0).sum(-1).astype(np.float32)
        a_mean = converted[1].mean(-1)
        a_l2_std = np.sqrt(
            converted[1][:, 0, :] ** 2 + converted[1][:, 1, :] ** 2 + converted[1][:, 2, :] ** 2).std(-1)[:,
                   np.newaxis]
        m_l2_std = np.sqrt(
            converted[2][:, 0, :] ** 2 + converted[2][:, 1, :] ** 2 + converted[2][:, 2, :] ** 2
        ).std(-1)[:, np.newaxis]

        features = np.concatenate((a_std, g_mean, g_std, m_over_0_count, a_mean, a_l2_std, m_l2_std), axis=1)
        labels = converted[0]  # onehot.fit_transform(converted[0].reshape(-1, 1)).toarray()

        length = labels.shape[0]

        classifier = LGBMClassifier()
        classifier.fit(features[:int(length * 0.7)], labels[:int(length * 0.7)])

        validate_y = labels[int(length * 0.7):]
        predict_y = classifier.predict(features[int(length * 0.7):])
        logger.info('Xiong')
        logger.info(f'Accuracy: {accuracy_score(validate_y, predict_y)}')
        logger.info(f'Precision: {precision_score(validate_y, predict_y, average=None)}')
        logger.info(f'Recall: {recall_score(validate_y, predict_y, average=None)}')

    def baseline_htc(self, profile: Profile, shared: Storage, logger: Logger, converted):
        a_std = converted[1].std(-1)
        a_mean = converted[1].mean(-1)
        m_std = converted[2].std(-1)
        m_mean = converted[2].mean(-1)
        g_std = converted[3].std(-1)
        g_mean = converted[3].mean(-1)
        a_fft = np.abs(np.fft.fft(converted[1], 256))[:, :, 1:128]
        a_fft_peak = a_fft.argmax(-1)
        a_fft_ratio = a_fft[:, :, 1] / a_fft[:, :, 0]
        m_fft = np.abs(np.fft.fft(converted[2], 256))[:, :, 1:128]
        m_fft_peak = m_fft.argmax(-1)
        m_fft_ratio = m_fft[:, :, 1] / m_fft[:, :, 0]
        g_fft = np.abs(np.fft.fft(converted[3], 256))[:, :, 1:128]
        g_fft_peak = g_fft.argmax(-1)
        g_fft_ratio = g_fft[:, :, 1] / g_fft[:, :, 0]

        features = np.concatenate((
            a_std, a_mean, a_fft_peak, a_fft_ratio,
            m_std, m_mean, m_fft_peak, m_fft_ratio,
            g_std, g_mean, g_fft_peak, g_fft_ratio,
        ), axis=1)
        features = robust_scale(features, axis=0)
        labels = converted[0]

        length = labels.shape[0]

        classifier = SVC()
        classifier.fit(features[:int(length * 0.7)], labels[:int(length * 0.7)])

        validate_y = labels[int(length * 0.7):]
        predict_y = classifier.predict(features[int(length * 0.7):])
        logger.info('HTC')
        logger.info(f'Accuracy: {accuracy_score(validate_y, predict_y)}')
        logger.info(f'Precision: {precision_score(validate_y, predict_y, average=None)}')
        logger.info(f'Recall: {recall_score(validate_y, predict_y, average=None)}')

    def baseline_lu(self, profile: Profile, shared: Storage, logger: Logger, converted):
        a_mean = converted[1].mean(-1)
        m_mean = converted[2].mean(-1)
        g_mean = converted[3].mean(-1)
        a_std = converted[1].std(-1)
        m_std = converted[2].std(-1)
        g_std = converted[3].std(-1)
        a_percentile = np.percentile(converted[1], [0, 25, 50, 75, 100], -1).transpose(1, 0, 2).reshape((-1, 15))
        m_percentile = np.percentile(converted[2], [0, 25, 50, 75, 100], -1).transpose(1, 0, 2).reshape((-1, 15))
        g_percentile = np.percentile(converted[3], [0, 25, 50, 75, 100], -1).transpose(1, 0, 2).reshape((-1, 15))
        a_mcr = (converted[1][:, :, -1] - converted[1][:, :, 0]) / 235
        m_mcr = (converted[2][:, :, -1] - converted[2][:, :, 0]) / 235
        g_mcr = (converted[3][:, :, -1] - converted[3][:, :, 0]) / 235

        a_fft = np.fft.fft(converted[1], 256, -1)[:, :, 1:128]
        m_fft = np.fft.fft(converted[2], 256, -1)[:, :, 1:128]
        g_fft = np.fft.fft(converted[3], 256, -1)[:, :, 1:128]

        a_df = a_fft.argmax(-1) * 47.0 / 512.0
        m_df = m_fft.argmax(-1) * 47.0 / 512.0
        g_df = g_fft.argmax(-1) * 47.0 / 512.0
        a_se = -(np.log2(np.abs(a_fft)) * np.abs(a_fft)).sum(-1)
        m_se = -(np.log2(np.abs(m_fft)) * np.abs(m_fft)).sum(-1)
        g_se = -(np.log2(np.abs(g_fft)) * np.abs(g_fft)).sum(-1)

        features = np.concatenate((
            a_mean, m_mean, g_mean, a_std, m_std, g_std, a_percentile, m_percentile, g_percentile,
            a_mcr, m_mcr, g_mcr, a_df, m_df, g_df, a_se, m_se, g_se
        ), axis=1)
        labels = converted[0]

        length = labels.shape[0]

        xgb_classifier = XGBClassifier()
        xgb_classifier.fit(features[:int(length * 0.7)], labels[:int(length * 0.7)])
        mlp_classifier = MLPClassifier()
        mlp_classifier.fit(features[:int(length * 0.7)], labels[:int(length * 0.7)])

        validate_y = labels[int(length * 0.7):]
        xgb_predict_p = xgb_classifier.predict_proba(features[int(length * 0.7):])
        mlp_predict_p = mlp_classifier.predict_proba(features[int(length * 0.7):])
        predict_y = (xgb_predict_p + mlp_predict_p).argmax(-1)
        logger.info('Lu')
        logger.info(f'Accuracy: {accuracy_score(validate_y, predict_y)}')
        logger.info(f'Precision: {precision_score(validate_y, predict_y, average=None)}')
        logger.info(f'Recall: {recall_score(validate_y, predict_y, average=None)}')

    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        dataset: HTCTMD = shared['loader'].dataset
        dataset.tensor = False
        sos = butter(8, 0.5, output='sos', fs=47)

        def convert_sensor(idx):
            if idx == 0:
                return np.concatenate(tuple(map(
                    lambda it: it[idx],
                    dataset
                )))
            else:
                return sosfilt(sos, np.concatenate(tuple(map(
                    lambda it: it[idx].reshape(1, *it[idx].shape),
                    dataset
                ))))

        converted_raw = tuple(map(
            lambda idx: convert_sensor(idx),
            range(4)
        ))

        # self.baseline_xiong(profile, shared, logger, converted_raw)
        self.baseline_htc(profile, shared, logger, converted_raw)
        # self.baseline_lu(profile, shared, logger, converted_raw)

        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        return ['loader']

    def provide(self) -> List[Text]:
        return []

    def remove(self) -> List[Text]:
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        return []


class DenseNet1dTrainTask(TemporalConvNetworkTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        return DenseNetLinear(profile)

    def define_model(cls):
        return []


class EmbraceNetTrainTask(TemporalConvNetworkTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        class Clasifier(nn.Module):
            def __init__(self):
                super(Clasifier, self).__init__()
                self.add_module('preprocess', nn.ModuleList(map(
                    lambda idx: nn.Sequential(
                        nn.Conv1d(7, 32, 3, stride=2),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(32, 64, 3, stride=2),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(64, 128, 3, stride=1),
                        nn.ReLU(inplace=True),
                        nn.Conv1d(128, 256, 3, stride=1),
                        nn.ReLU(inplace=True),
                        nn.Flatten()
                    ),
                    range(3)
                )))
                self.add_module('embrace', EmbraceNet(torch.device('cuda:0'), [256 * 54] * 3, 2048))
                self.add_module('postprocess', nn.Sequential(
                    nn.Conv1d(2048, 1024, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(1024, 256, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(256, 8, 1),
                    nn.Flatten()
                ))

            def forward(self, x):
                x_a, x_m, x_g = x
                x_a = self.preprocess[0](x_a)
                x_m = self.preprocess[1](x_m)
                x_g = self.preprocess[2](x_g)
                x = self.embrace((x_a, x_m, x_g))
                x = x.unsqueeze(-1)
                return self.postprocess(x)

        return Clasifier()

    def define_model(cls):
        return []


class OrangeLabsTrainTask(TemporalConvNetworkTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        class Classifier(nn.Module):
            def __init__(self):
                super(Classifier, self).__init__()
                self.add_module('flatten', nn.Flatten())
                self.add_module('bilstm', nn.ModuleList(map(
                    lambda idx: nn.LSTM(47 * 7, 100, 2, bidirectional=True),
                    range(3)
                )))
                self.add_module('linear', nn.Linear(400 * 3, 8))

            def forward(self, x):
                x_a, x_m, x_g = x
                x_a = x_a.view(*x_a.shape[:-1], 5, -1).transpose(2, 0).transpose(1, 2).flatten(-2)
                x_m = x_m.view(*x_m.shape[:-1], 5, -1).transpose(2, 0).transpose(1, 2).flatten(-2)
                x_g = x_g.view(*x_g.shape[:-1], 5, -1).transpose(2, 0).transpose(1, 2).flatten(-2)
                x_a = self.bilstm[0](x_a)[1][0].transpose(0, 1).flatten(-2)
                x_m = self.bilstm[0](x_m)[1][0].transpose(0, 1).flatten(-2)
                x_g = self.bilstm[0](x_g)[1][0].transpose(0, 1).flatten(-2)
                x = torch.cat((x_a, x_m, x_g), dim=1)
                x = self.linear(x)
                return x

        return Classifier()

    def define_model(cls):
        return []
