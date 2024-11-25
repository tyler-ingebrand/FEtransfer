from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder import FunctionEncoder, BaseDataset
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback

from src.algs.MAML import MAML
from src.algs.Oracle import Oracle
from src.algs.PrototypicalNetwork import ProtoTypicalNetwork
from src.algs.generic_function_space_methods import _distance


class CustomCallback(BaseCallback):

    def __init__(self,
                 data_type:str,
                 testing_dataset:Union[BaseDataset, type(None)],
                 logdir: Union[str, None] = None,
                 tensorboard: Union[None, SummaryWriter] = None,
                 prefix="type1",
                 ):
        """ Constructor for MSECallback. Either logdir  or tensorboard must be provided, but not both"""
        assert logdir is not None or tensorboard is not None, "Either logdir or tensorboard must be provided"
        assert logdir is None or tensorboard is None, "Only one of logdir or tensorboard can be provided"
        super(CustomCallback, self).__init__()
        self.testing_dataset = testing_dataset
        if logdir is not None:
            self.tensorboard = SummaryWriter(logdir)
        else:
            self.tensorboard = tensorboard
        self.prefix = prefix
        self.total_epochs = 0
        self.data_type = data_type

    def on_training_start(self, locals: dict):
        if self.prefix == "type2" and self.data_type == "categorical":
            return # skip type2 for categorical data, since a linear combination of distributions does not happen very often.
        if self.total_epochs == 0 and self.testing_dataset is not None:
            # with torch.no_grad():
                # get model and data
                model = locals["self"]
                example_xs, example_ys, xs, ys, info = self.testing_dataset.sample()

                # test and log
                to_log = self.eval(model, example_xs, example_ys, xs, ys, info)
                for tag, value in to_log.items():
                    self.tensorboard.add_scalar(tag, value, self.total_epochs)
                self.total_epochs += 1

    def on_step(self, locals:dict):
        if self.prefix == "type2" and self.data_type == "categorical":
            return # skip type2 for categorical data, since a linear combination of distributions does not happen very often.

        # with torch.no_grad():
        model = locals["self"]

        if (locals['epoch'] + 1) % model.gradient_accumulation == 0:

            # sample testing data
            if self.testing_dataset is not None:
                example_xs, example_ys, query_xs, query_ys, info = self.testing_dataset.sample()
            else:
                example_xs, example_ys, query_xs, query_ys, info = locals["example_xs"], locals["example_ys"], locals["query_xs"], locals["query_ys"], locals["_"]

            # test
            to_log = self.eval(model, example_xs, example_ys, query_xs, query_ys, info)

            # log results
            for tag, value in to_log.items():
                self.tensorboard.add_scalar(tag, value, self.total_epochs)
            self.total_epochs += 1



    def eval(self, model, example_xs, example_ys, xs, ys, info):
        # NOTE: MAML needs grads.
        if not isinstance(model, MAML):
            model.eval()

        # eval models
        if type(model) == Oracle or type(model) == ProtoTypicalNetwork:
            y_hats = model.predict_from_examples(example_xs, example_ys, xs, info=info)
        else:
            y_hats = model.predict_from_examples(example_xs, example_ys, xs, method=model.method if isinstance(model, FunctionEncoder) else None)

        # reenable grads and dropout
        if not isinstance(model, MAML):
            model.train()

        # measure mse
        if self.data_type == "deterministic":
            loss = _distance(y_hats, ys, data_type=self.data_type, squared=True).mean()
            return {f"{self.prefix}/mean_distance_squared": loss}
        elif self.data_type == "stochastic":
            loss = _distance(y_hats, ys, data_type=self.data_type, squared=True).mean()
            return {f"{self.prefix}/mean_distance_squared": loss}
        elif self.data_type == "categorical":
            # distance loss
            loss = _distance(y_hats, ys, data_type=self.data_type, squared=True).mean()

            # accuracy - What fraction of the time the probability of the correct class is greater than the probability of the incorrect class
            true_class = ys.argmax(dim=-1)
            estimated_class = y_hats.argmax(dim=-1)
            accuracy = (true_class == estimated_class).float().mean()

            # cross entropy
            cross_entropy = torch.nn.CrossEntropyLoss()(y_hats.reshape(-1, 2), true_class.reshape(-1))

            return {f"{self.prefix}/mean_distance_squared": loss,
                    f"{self.prefix}/accuracy": accuracy,
                    f"{self.prefix}/cross_entropy": cross_entropy
                    }
        else:
            raise NotImplementedError("Only deterministic data is supported")
