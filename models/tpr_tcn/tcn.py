from logging import Logger, getLogger
from multiprocessing.dummy import Pool
from os import makedirs, path
from typing import OrderedDict, List, Text

import torch
from ignite import metrics, engine
from torch import nn
from waterch.tasker import Profile, value, Definition, Return
from waterch.tasker.storage import Storage
from waterch.tasker.tasks import Task
from waterch.tasker.tasks.torch import SimpleTrainTask
from waterch.tasker.utils import import_reference

from models.resnet1d import ResNet1d
from models.tcn import TemporalConvNet


class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.add_module('channel_wise', nn.AdaptiveAvgPool1d(1))

    def forward(self, x):
        children = dict(self.named_children())
        return x * children['channel_wise'](x).softmax(-1)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(self.dim)


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.squeeze()


class TemporalConvNetworkTrainTask(SimpleTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        activation = import_reference(profile.activation_reference)

        class Classifier(nn.Module):
            def __init__(self):
                super(Classifier, self).__init__()
                self.add_module('resnets', nn.ModuleList(map(
                    lambda it: ResNet1d(**it, activation=activation),
                    profile.resnets
                )))

                self.add_module('tcns', nn.ModuleList(map(
                    lambda it: TemporalConvNet(**it, activation=activation),
                    profile.tcns
                )))

                self.add_module('flatten', nn.Sequential(
                    SelfAttention(),
                    nn.Flatten(),
                    Unsqueeze(1),
                    nn.AdaptiveAvgPool1d(profile.pool_size),
                    Squeeze()
                ))

                self.add_module('linears', nn.Sequential(*tuple(map(
                    lambda idx: nn.Sequential(
                        nn.Linear(
                            profile.pool_size
                            if idx == 0
                            else profile.linear_features[idx - 1],
                            profile.num_classes
                            if idx == len(profile.linear_features) - 1
                            else profile.linear_features[idx],
                            True
                        ),
                        activation(True) if idx != len(profile.linear_features) - 1 else nn.Sequential()
                    ),
                    range(len(profile.linear_features))
                ))) if len(profile.linear_features) > 0 else nn.Linear(
                    profile.pool_size, profile.num_classes, True
                ))

                self.init_weights()

            def init_weights(self):
                for module in self.modules():
                    if isinstance(module, nn.Conv1d):
                        module.weight.data.normal_(0, 1)
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(0, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                children = dict(self.named_modules())
                pool = Pool(len(x))
                y_pred = children['linears'](children['flatten'](torch.cat(tuple(pool.map(
                    lambda idx: children['tcns'][idx](children['resnets'][idx](x[idx])),
                    range(len(x))
                )), dim=1)))
                return y_pred

        return Classifier()

    @classmethod
    def define_model(cls):
        return [
            value('activation_reference', str),
            value('pool_size', int),
            value('linear_features', [int]),
            value('num_classes', int),
            value('tcns', list, [
                [
                    value('in_channels', int),
                    value('layer_channels', list, [int]),
                    value('kernel_size', int),
                    value('dropout', float)
                ]
            ]),
            value('resnets', list, [
                ResNet1d.define()
            ])
        ]

    def prepare_batch(self, batch, device=None, non_blocking=False):
        def components(tensor: torch.Tensor):
            batch_size, num_channels, num_features = tensor.shape
            if num_channels == 3:
                x = tensor[:, 0, :].reshape(-1)
                y = tensor[:, 1, :].reshape(-1)
                z = tensor[:, 2, :].reshape(-1)
                f = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
                f[torch.isnan(f)] = 0.0
                h = torch.sqrt(x ** 2 + y ** 2)
                h[torch.isnan(h)] = 0.0
                i = torch.asin(z / f)
                i[torch.isnan(i)] = 0.0
                d = torch.asin(y / h)
                d[torch.isnan(d)] = 0.0
                return torch.cat(tuple(map(
                    lambda it: it.reshape(batch_size, 1, num_features),
                    (x, y, z, f, h, i, d)
                )), dim=1)
            else:
                return batch

        y = batch.pop(0)
        if len(y.shape) == 3:
            y = engine.convert_tensor(y[:, :, 0].reshape(-1), device, non_blocking)
        elif len(y.shape) == 2:
            y = engine.convert_tensor(y[:, 0], device, non_blocking)
        else:
            y = engine.convert_tensor(y, device, non_blocking)
        x = tuple(map(
            lambda it: engine.convert_tensor(components(it[1]), device, non_blocking),
            sorted(batch.items(), key=lambda it: it[0])
        ))
        return x, y

    def more_metrics(self, metrics_: OrderedDict):
        metrics_['loss'] = metrics.Loss(nn.CrossEntropyLoss())
        metrics_['accuracy'] = metrics.Accuracy()
        metrics_['recall'] = metrics.Recall()
        metrics_['precision'] = metrics.Precision()
        metrics_['confusion_matrix'] = metrics.ConfusionMatrix(8, average='recall')

    def create_trainer(
            self, model, optimizer, loss_fn, device, non_blocking, prepare_batch,
            output_transform=lambda x, y, y_pred, loss: loss.item()
    ):
        logger = getLogger(f'models.tpr_tcn.tcn.TemporalConvNetworkTrainTask[{id(self)}]::create_trainer')

        def output_transform(x, y, y_pred, loss):
            logger.debug(y_pred)
            return loss.item()

        return super(TemporalConvNetworkTrainTask, self).create_trainer(model, optimizer, loss_fn, device, non_blocking,
                                                                        prepare_batch, output_transform)


class TemporalConvNetworkAttentionPlotTask(Task):
    def invoke(self, profile: Profile, shared: Storage, logger: Logger) -> int:
        try:
            makedirs(profile.cache_dir)
        except FileExistsError:
            logger.warning('Plot cache folder already exists')
        model = shared['model']
        validate_loader = shared['validate_loader']

        self_attention: nn.Module = model.flatten[0]

        def prepare_batch(batch, device=None, non_blocking=False):
            def components(tensor: torch.Tensor):
                batch_size, num_channels, num_features = tensor.shape
                if num_channels == 3:
                    x = tensor[:, 0, :].reshape(-1)
                    y = tensor[:, 1, :].reshape(-1)
                    z = tensor[:, 2, :].reshape(-1)
                    f = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
                    f[torch.isnan(f)] = 0.0
                    h = torch.sqrt(x ** 2 + y ** 2)
                    h[torch.isnan(h)] = 0.0
                    i = torch.asin(z / f)
                    i[torch.isnan(i)] = 0.0
                    d = torch.asin(y / h)
                    d[torch.isnan(d)] = 0.0
                    return torch.cat(tuple(map(
                        lambda it: it.reshape(batch_size, 1, num_features),
                        (x, y, z, f, h, i, d)
                    )), dim=1)
                else:
                    return batch

            y = batch.pop(0)
            if len(y.shape) == 3:
                y = engine.convert_tensor(y[:, :, 0].reshape(-1), device, non_blocking)
            elif len(y.shape) == 2:
                y = engine.convert_tensor(y[:, 0], device, non_blocking)
            else:
                y = engine.convert_tensor(y, device, non_blocking)
            x = tuple(map(
                lambda it: engine.convert_tensor(components(it[1]), device, non_blocking),
                sorted(batch.items(), key=lambda it: it[0])
            ))
            return x, y

        inputs = []
        outputs = []
        labels = []
        for batch in validate_loader:
            x, y = prepare_batch(batch, torch.device(profile.device), True)

            def attention_hook(module, input: torch.Tensor, output: torch.Tensor):
                inputs.append(input)
                outputs.append(output)
                labels.append(y)

            handler = self_attention.register_forward_hook(attention_hook)
            with torch.no_grad():
                model(x)

            handler.remove()

        torch.save({'inputs': inputs, 'outputs': outputs, 'labels': labels},
                   path.join(profile.cache_dir, 'attention_hooked.pth'))
        return Return.SUCCESS.value

    def require(self) -> List[Text]:
        return ['model', 'validate_loader']

    def provide(self) -> List[Text]:
        return []

    def remove(self) -> List[Text]:
        return []

    @classmethod
    def define(cls) -> List[Definition]:
        return [
            value('cache_dir', str),
            value('device', str),
        ]


class TemporalConvNetworkNoResNetTrainTask(TemporalConvNetworkTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        activation = import_reference(profile.activation_reference)

        class Classifier(nn.Module):
            def __init__(self):
                super(Classifier, self).__init__()
                self.add_module('resnets', nn.ModuleList(map(
                    lambda it: nn.Sequential(),  # ResNet1d(**it, activation=activation),
                    profile.resnets
                )))

                self.add_module('tcns', nn.ModuleList(map(
                    lambda it: TemporalConvNet(**it, activation=activation),
                    profile.tcns
                )))

                self.add_module('flatten', nn.Sequential(
                    SelfAttention(),
                    nn.Flatten(),
                    Unsqueeze(1),
                    nn.AdaptiveAvgPool1d(profile.pool_size),
                    Squeeze()
                ))

                self.add_module('linears', nn.Sequential(*tuple(map(
                    lambda idx: nn.Sequential(
                        nn.Linear(
                            profile.pool_size
                            if idx == 0
                            else profile.linear_features[idx - 1],
                            profile.num_classes
                            if idx == len(profile.linear_features) - 1
                            else profile.linear_features[idx],
                            True
                        ),
                        activation(True) if idx != len(profile.linear_features) - 1 else nn.Sequential()
                    ),
                    range(len(profile.linear_features))
                ))) if len(profile.linear_features) > 0 else nn.Linear(
                    profile.pool_size, profile.num_classes, True
                ))

                self.init_weights()

            def init_weights(self):
                for module in self.modules():
                    if isinstance(module, nn.Conv1d):
                        module.weight.data.normal_(0, 1)
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(0, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                children = dict(self.named_modules())
                pool = Pool(len(x))
                y_pred = children['linears'](children['flatten'](torch.cat(tuple(pool.map(
                    lambda idx: children['tcns'][idx](children['resnets'][idx](x[idx])),
                    range(len(x))
                )), dim=1)))
                return y_pred

        return Classifier()


class TemporalConvNetworkNoTCNTrainTask(TemporalConvNetworkTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        activation = import_reference(profile.activation_reference)

        class Classifier(nn.Module):
            def __init__(self):
                super(Classifier, self).__init__()
                self.add_module('resnets', nn.ModuleList(map(
                    lambda it: ResNet1d(**it, activation=activation),
                    profile.resnets
                )))

                self.add_module('tcns', nn.ModuleList(map(
                    lambda it: nn.Sequential(),  # TemporalConvNet(**it, activation=activation),
                    profile.tcns
                )))

                self.add_module('flatten', nn.Sequential(
                    SelfAttention(),
                    nn.Flatten(),
                    Unsqueeze(1),
                    nn.AdaptiveAvgPool1d(profile.pool_size),
                    Squeeze()
                ))

                self.add_module('linears', nn.Sequential(*tuple(map(
                    lambda idx: nn.Sequential(
                        nn.Linear(
                            profile.pool_size
                            if idx == 0
                            else profile.linear_features[idx - 1],
                            profile.num_classes
                            if idx == len(profile.linear_features) - 1
                            else profile.linear_features[idx],
                            True
                        ),
                        activation(True) if idx != len(profile.linear_features) - 1 else nn.Sequential()
                    ),
                    range(len(profile.linear_features))
                ))) if len(profile.linear_features) > 0 else nn.Linear(
                    profile.pool_size, profile.num_classes, True
                ))

                self.init_weights()

            def init_weights(self):
                for module in self.modules():
                    if isinstance(module, nn.Conv1d):
                        module.weight.data.normal_(0, 1)
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(0, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                children = dict(self.named_modules())
                pool = Pool(len(x))
                y_pred = children['linears'](children['flatten'](torch.cat(tuple(pool.map(
                    lambda idx: children['tcns'][idx](children['resnets'][idx](x[idx])),
                    range(len(x))
                )), dim=1)))
                return y_pred

        return Classifier()


class TemporalConvNetworkNoCNNTrainTask(TemporalConvNetworkTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        activation = import_reference(profile.activation_reference)

        class Classifier(nn.Module):
            def __init__(self):
                super(Classifier, self).__init__()
                self.add_module('resnets', nn.ModuleList(map(
                    lambda it: nn.Sequential(),  # ResNet1d(**it, activation=activation),
                    profile.resnets
                )))

                self.add_module('tcns', nn.ModuleList(map(
                    lambda it: nn.Sequential(),  # TemporalConvNet(**it, activation=activation),
                    profile.tcns
                )))

                self.add_module('flatten', nn.Sequential(
                    SelfAttention(),
                    nn.Flatten(),
                    Unsqueeze(1),
                    nn.AdaptiveAvgPool1d(profile.pool_size),
                    Squeeze()
                ))

                self.add_module('linears', nn.Sequential(*tuple(map(
                    lambda idx: nn.Sequential(
                        nn.Linear(
                            profile.pool_size
                            if idx == 0
                            else profile.linear_features[idx - 1],
                            profile.num_classes
                            if idx == len(profile.linear_features) - 1
                            else profile.linear_features[idx],
                            True
                        ),
                        activation(True) if idx != len(profile.linear_features) - 1 else nn.Sequential()
                    ),
                    range(len(profile.linear_features))
                ))) if len(profile.linear_features) > 0 else nn.Linear(
                    profile.pool_size, profile.num_classes, True
                ))

                self.init_weights()

            def init_weights(self):
                for module in self.modules():
                    if isinstance(module, nn.Conv1d):
                        module.weight.data.normal_(0, 1)
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(0, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                children = dict(self.named_modules())
                pool = Pool(len(x))
                y_pred = children['linears'](children['flatten'](torch.cat(tuple(pool.map(
                    lambda idx: children['tcns'][idx](children['resnets'][idx](x[idx])),
                    range(len(x))
                )), dim=1)))
                return y_pred

        return Classifier()


class TemporalConvNetworkNoAttentionTrainTask(TemporalConvNetworkTrainTask):
    def create_model(self, profile: Profile, shared: Storage, logger: Logger) -> nn.Module:
        activation = import_reference(profile.activation_reference)

        class Classifier(nn.Module):
            def __init__(self):
                super(Classifier, self).__init__()
                self.add_module('resnets', nn.ModuleList(map(
                    lambda it: ResNet1d(**it, activation=activation),
                    profile.resnets
                )))

                self.add_module('tcns', nn.ModuleList(map(
                    lambda it: TemporalConvNet(**it, activation=activation),
                    profile.tcns
                )))

                self.add_module('flatten', nn.Sequential(
                    nn.Flatten(),
                    Unsqueeze(1),
                    nn.AdaptiveAvgPool1d(profile.pool_size),
                    Squeeze()
                ))

                self.add_module('linears', nn.Sequential(*tuple(map(
                    lambda idx: nn.Sequential(
                        nn.Linear(
                            profile.pool_size
                            if idx == 0
                            else profile.linear_features[idx - 1],
                            profile.num_classes
                            if idx == len(profile.linear_features) - 1
                            else profile.linear_features[idx],
                            True
                        ),
                        activation(True) if idx != len(profile.linear_features) - 1 else nn.Sequential()
                    ),
                    range(len(profile.linear_features))
                ))) if len(profile.linear_features) > 0 else nn.Linear(
                    profile.pool_size, profile.num_classes, True
                ))

                self.init_weights()

            def init_weights(self):
                for module in self.modules():
                    if isinstance(module, nn.Conv1d):
                        module.weight.data.normal_(0, 1)
                    if isinstance(module, nn.Linear):
                        module.weight.data.normal_(0, 1)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                children = dict(self.named_modules())
                pool = Pool(len(x))
                y_pred = children['linears'](children['flatten'](torch.cat(tuple(pool.map(
                    lambda idx: children['tcns'][idx](children['resnets'][idx](x[idx])),
                    range(len(x))
                )), dim=1)))
                return y_pred

        return Classifier()
