from typing import Union

import torch
from FunctionEncoder import BaseCallback, BaseDataset
from FunctionEncoder.Model.Architecture.CNN import CNN
from FunctionEncoder.Model.Architecture.MLP import MLP, get_activation
from tqdm import trange

from src.algs.BaseAlg import BaseAlg
from src.algs.generic_function_space_methods import _distance


class BruteForceBasis(BaseAlg):
    
    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs):
        n_params = 0
        if model_type == "CNN":
            n_params += CNN.predict_number_params(input_size=input_size, output_size=(model_kwargs["hidden_size"],), n_basis=1, n_layers=2,  learn_basis_functions=False, hidden_size=model_kwargs["hidden_size"])
            ins = model_kwargs["hidden_size"]
        else:
            ins = input_size[0]
        num_examples = model_kwargs["n_examples"]
        ins_example = (ins + output_size[0]) * num_examples
        outs_example = n_basis
        ins_basis = ins
        outs_basis = output_size[0]
        n_params += MLP.predict_number_params((ins_example,), (outs_example,), n_basis=1,  learn_basis_functions=False, **model_kwargs)
        n_params += MLP.predict_number_params((ins_basis,), (outs_basis,), n_basis=n_basis, **model_kwargs)

        return n_params
    
    def __init__(self,
                 input_size :tuple[int],
                 output_size :tuple[int],
                 data_type :str,
                 n_basis:int,
                 num_data :int,
                 model_type :Union[str, type] ="MLP",
                 model_kwargs :dict =dict(),
                 gradient_accumulation :int =1,
                 cross_entropy: bool = False,
                 ):
        super(BruteForceBasis, self).__init__(input_size=input_size, output_size=output_size, data_type=data_type,
                                          n_basis=n_basis, model_type=model_type, model_kwargs=model_kwargs,
                                          gradient_accumulation=gradient_accumulation, cross_entropy=cross_entropy)

        if self.model_type == "CNN":
            self.conv = CNN(input_size=input_size, output_size=(model_kwargs["hidden_size"],), n_basis=1,  learn_basis_functions=False, n_layers=2, hidden_size=model_kwargs["hidden_size"])
            ins = model_kwargs["hidden_size"]
        else:
            ins = input_size[0]
        # model details
        ins_example = (ins + output_size[0]) * num_data
        outs_example = n_basis
        ins_basis = ins
        outs_basis = output_size[0]

        # create example to basis model
        self.example_to_basis = MLP(input_size=(ins_example,), output_size=(outs_example,), n_basis=1, learn_basis_functions=False, **model_kwargs)

        # create basis
        self.basis = MLP(input_size=(ins_basis,), output_size=(outs_basis,), n_basis=n_basis, **model_kwargs)

        # opti
        params = [*self.example_to_basis.parameters()] + [*self.basis.parameters()]
        if model_type == "CNN":
            params += [*self.conv.parameters()]
        self.opt = torch.optim.Adam(params, lr=1e-3)


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
        examples = examples.reshape(examples.shape[0], -1) # f x (n+m)*e

        # pass through example_to_basis
        basis_coefficients = self.example_to_basis(examples) # f x k

        # pass through basis
        basis_functions = self.basis(xs) # f x d x m x k

        # compute linear combination
        y_hats = torch.einsum("fdmk,fk->fdm", basis_functions, basis_coefficients)
        return y_hats

