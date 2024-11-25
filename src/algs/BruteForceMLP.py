from typing import Union

import torch
from FunctionEncoder import BaseCallback, BaseDataset
from FunctionEncoder.Model.Architecture.CNN import CNN
from FunctionEncoder.Model.Architecture.MLP import MLP, get_activation
from tqdm import trange

from src.algs.BaseAlg import BaseAlg
from src.algs.generic_function_space_methods import _distance


class BruteForceMLP(BaseAlg):
    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs):
        num_examples = model_kwargs["n_examples"]

        n_params = 0
        if model_type == "CNN":
            n_params += CNN.predict_number_params(input_size=input_size, output_size=(model_kwargs["hidden_size"],), n_basis=1, n_layers=2,  learn_basis_functions=False,  hidden_size=model_kwargs["hidden_size"])
            ins = model_kwargs["hidden_size"]
        else:
            ins = input_size[0]

        ins = (ins + output_size[0]) * num_examples + ins
        outs = output_size[0]
        n_params += MLP.predict_number_params((ins,), (outs,), n_basis=1,  learn_basis_functions=False,  **model_kwargs)

        return n_params



    def __init__(self,
                 input_size :tuple[int],
                 output_size :tuple[int],
                 data_type :str,
                 num_data :int,
                 model_type :Union[str, type] ="MLP",
                 model_kwargs :dict =dict(),
                 gradient_accumulation :int =1,
                 cross_entropy: bool = False,
                 ):

        super(BruteForceMLP, self).__init__(input_size=input_size, output_size=output_size, data_type=data_type,
                                          n_basis=1, model_type=model_type, model_kwargs=model_kwargs,
                                          gradient_accumulation=gradient_accumulation, cross_entropy=cross_entropy)

        if self.model_type == "CNN":
            self.conv = CNN(input_size=input_size, output_size=(model_kwargs["hidden_size"],), n_basis=1, n_layers=2, learn_basis_functions=False,  hidden_size=model_kwargs["hidden_size"])
            ins = model_kwargs["hidden_size"]
        else:
            ins = input_size[0]

        # model details
        ins = (ins + output_size[0]) * num_data + ins
        outs = output_size[0]

        # create model and opt
        self.model = MLP((ins,), (outs,), n_basis=1,  learn_basis_functions=False, **model_kwargs)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def predict_from_examples(self,
                              example_xs: torch.tensor,
                              example_ys: torch.tensor,
                              xs: torch.tensor, # f x d x n
                              **kwargs):
        # convert images to vectors
        if self.model_type == "CNN":
            example_xs = self.conv(example_xs)
            xs = self.conv(xs)

        # create input as all examples and the inputs to predict
        examples = torch.cat([example_xs, example_ys], dim=-1) # f x e x n+m
        examples = examples.reshape(examples.shape[0], 1, -1) # f x 1 x (n+m)*e
        examples = examples.expand(-1, xs.shape[1], -1) # f x d x (n+m)*e
        inputs = torch.cat([examples, xs], dim=-1) # f x d x n+(n+m)*e

        # pass through nn
        y_hats = self.model(inputs)
        return y_hats
