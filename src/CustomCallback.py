from typing import Union

import torch
from torch.utils.tensorboard import SummaryWriter

from FunctionEncoder import FunctionEncoder, BaseDataset
from FunctionEncoder.Callbacks.BaseCallback import BaseCallback

from src.algs.MAML import MAML
from src.algs.Oracle import Oracle


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
        if self.total_epochs == 0 and self.testing_dataset is not None:
            # with torch.no_grad():
                # get model and data
                model = locals["self"]
                example_xs, example_ys, xs, ys, info = self.testing_dataset.sample()

                # test and log
                tag, loss = self.eval(model, example_xs, example_ys, xs, ys, info)
                self.tensorboard.add_scalar(tag, loss, self.total_epochs)
                self.total_epochs += 1

    def on_step(self, locals:dict):
        # with torch.no_grad():
            model = locals["self"]

            # sample testing data
            if self.testing_dataset is not None:
                example_xs, example_ys, xs, ys, info = self.testing_dataset.sample()
            else:
                example_xs, example_ys, xs, ys, info = locals["example_xs"], locals["example_ys"], locals["xs"], locals["ys"], locals["_"]

            # test
            tag, loss = self.eval(model, example_xs, example_ys, xs, ys, info)

            # log results
            self.tensorboard.add_scalar(tag, loss, self.total_epochs)
            self.total_epochs += 1



    def eval(self, model, example_xs, example_ys, xs, ys, info):
        # NOTE: MAML needs grads.
        if not isinstance(model, MAML):
            model.eval()

        # eval models
        if type(model) == Oracle:
            y_hats = model.predict_from_examples(example_xs, example_ys, xs, info=info, method="least_squares")
        else:
            y_hats = model.predict_from_examples(example_xs, example_ys, xs, method="least_squares")

        # reenable grads and dropout
        if not isinstance(model, MAML):
            model.train()

        # measure mse
        if self.data_type == "deterministic":
            loss = model._distance(y_hats, ys, squared=True).mean()
            return f"{self.prefix}/mean_distance_squared", loss
        elif self.data_type == "stochastic":
            loss = model._distance(y_hats, ys, squared=True).mean()
            return f"{self.prefix}/mean_distance_squared", loss
        else:
            raise NotImplementedError("Only deterministic data is supported")
