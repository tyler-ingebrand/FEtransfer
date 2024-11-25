from typing import Union

import torch
from FunctionEncoder import BaseCallback, BaseDataset
from FunctionEncoder.Model.Architecture.CNN import CNN
from FunctionEncoder.Model.Architecture.MLP import MLP
from tqdm import trange

from src.algs.BaseAlg import BaseAlg
from src.algs.generic_function_space_methods import _distance


class Oracle(BaseAlg):

    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs):
        oracle_size = model_kwargs["oracle_size"]
        n_params = 0
        if model_type == "CNN":
            n_params += CNN.predict_number_params(input_size=input_size, output_size=(model_kwargs["hidden_size"],), n_basis=1, n_layers=2,  learn_basis_functions=False, hidden_size=model_kwargs["hidden_size"])
            ins = model_kwargs["hidden_size"]
        else:
            ins = input_size[0]
        ins = ins + oracle_size
        outs = output_size[0]
        n_params += MLP.predict_number_params((ins,), (outs,), n_basis=1,  learn_basis_functions=False, **model_kwargs)
        return n_params

    def __init__(self,
                 input_size :tuple[int],
                 output_size :tuple[int],
                 oracle_size:int,
                 data_type :str,
                 model_type :Union[str, type],
                 model_kwargs :dict =dict(),
                 gradient_accumulation :int =1,
                 cross_entropy: bool = False,
                 ):
        """ Initializes a function encoder.

        Args:
        input_size: tuple[int]: The size of the input space, e.g. (1,) for 1D input
        output_size: tuple[int]: The size of the output space, e.g. (1,) for 1D output
        data_type: str: "deterministic" or "stochastic". Determines which defintion of inner product is used.
        n_basis: int: Number of basis functions to use.
        model_type: str: The type of model to use. See the types and kwargs in FunctionEncoder/Model/Architecture. Typically a MLP.
        model_kwargs: Union[dict, type(None)]: The kwargs to pass to the model. See the types and kwargs in FunctionEncoder/Model/Architecture.
        gradient_accumulation: int: The number of batches to accumulate gradients over. Typically its best to have n_functions>=10 or so, and have gradient_accumulation=1. However, sometimes due to memory reasons, or because the functions do not have the same amount of data, its necesary for n_functions=1 and gradient_accumulation>=10.
        """


        super(Oracle, self).__init__(input_size=input_size, output_size=output_size, data_type=data_type,
                                          n_basis=1, model_type=model_type, model_kwargs=model_kwargs,
                                          gradient_accumulation=gradient_accumulation, cross_entropy=cross_entropy)
        if self.model_type == "CNN":
            self.conv = CNN(input_size=input_size, output_size=(model_kwargs["hidden_size"],), n_basis=1, n_layers=2, learn_basis_functions=False,  hidden_size=model_kwargs["hidden_size"])
            ins = model_kwargs["hidden_size"]
        else:
            ins = input_size[0]

        # models and optimizers
        ins = ins + oracle_size
        outs = output_size[0]
        self.model = MLP((ins,), (outs,), n_basis=1,  learn_basis_functions=False, **model_kwargs)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)


    def predict_from_examples(self,
                              example_xs: torch.tensor,
                              example_ys: torch.tensor,
                              xs: torch.tensor,
                              info: dict,
                              **kwargs):
        # get priveleged information
        oracle_inputs = info["oracle_inputs"]
        if len(oracle_inputs.shape) == 2:
            oracle_inputs = oracle_inputs.unsqueeze(1).expand(-1, xs.shape[1], -1)

        # convert images to vectors
        if self.model_type == "CNN":
            xs = self.conv(xs)


        inputs = torch.cat((xs, oracle_inputs), dim=-1)
        y_hats = self.model(inputs)
        return y_hats

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
            y_hats = self.predict_from_examples(example_xs, example_ys, query_xs, _) # note this is special because it needs privelged info
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
