from typing import Union

import torch
from FunctionEncoder import BaseCallback, BaseDataset
from FunctionEncoder.Model.Architecture.CNN import CNN
from FunctionEncoder.Model.Architecture.MLP import MLP
from tqdm import trange

from src.algs.generic_function_space_methods import _distance
class BaseAlg(torch.nn.Module):

    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs):
        raise NotImplementedError

    def __init__(self,
                 input_size :tuple[int],
                 output_size :tuple[int],
                 data_type :str,
                 n_basis :int =100,
                 model_type :Union[str, type] ="MLP",
                 model_kwargs :dict =dict(),
                 gradient_accumulation :int =1,
                 cross_entropy=False,
                 ):
        if model_type == "MLP":
            assert len(input_size) == 1, "Only 1D input supported for MLPs. If your input is an image, set 'model_type' to 'CNN'"
        if model_type == "CNN":
            assert len(input_size) == 3, "Only 3D input supported for CNNs. If your input is a 1D vector, set 'model_type' to 'MLP'"
        assert input_size[0] >= 1, "Input size must be at least 1"
        assert len(output_size) == 1, "Only 1D output supported for now"
        assert output_size[0] >= 1, "Output size must be at least 1"
        assert data_type in ["deterministic", "stochastic", "categorical"], f"Unknown data type: {data_type}"
        super(BaseAlg, self).__init__()

        # hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = n_basis
        self.data_type = data_type
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.gradient_accumulation = gradient_accumulation
        self.cross_entropy = cross_entropy

    def predict_from_examples(self,
                              example_xs: torch.tensor,
                              example_ys: torch.tensor,
                              xs: torch.tensor,
                              **kwargs):
        raise NotImplementedError

    def train_model(self,
                    dataset: BaseDataset,
                    epochs: int,
                    progress_bar=True,
                    callback: BaseCallback = None):
        # Let callbacks few starting data
        if callback is not None:
            callback.on_training_start(locals())

        losses = []
        bar = trange(epochs) if progress_bar else range(epochs)
        for epoch in bar:
            example_xs, example_ys, query_xs, query_ys, _ = dataset.sample()

            # approximate functions, compute error
            y_hats = self.predict_from_examples(example_xs, example_ys, query_xs)
            if not self.cross_entropy or self.data_type != "categorical":
                loss = _distance(y_hats, query_ys, data_type=self.data_type, squared=True).mean()
            else:
                classes = query_ys.argmax(dim=2)
                loss = torch.nn.CrossEntropyLoss()(y_hats.reshape(-1, 2), classes.reshape(-1))

            # backprop with gradient clipping
            loss.backward()
            if (epoch + 1) % self.gradient_accumulation == 0:
                norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.opt.step()
                self.opt.zero_grad()

            # callbacks
            if callback is not None:
                callback.on_step(locals())

        # let callbacks know its done
        if callback is not None:
            callback.on_training_end(locals())
