from typing import Union

import torch
from FunctionEncoder import BaseCallback, BaseDataset
from FunctionEncoder.Model.Architecture.CNN import CNN
from FunctionEncoder.Model.Architecture.MLP import MLP
from tqdm import trange

from src.algs.BaseAlg import BaseAlg
from src.algs.generic_function_space_methods import _distance
class AutoEncoder(BaseAlg):

    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs):
        n_params = 0
        if model_type == "CNN":
            n_params += CNN.predict_number_params(input_size=input_size, output_size=(model_kwargs["hidden_size"],), n_basis=1, n_layers=2, hidden_size=model_kwargs["hidden_size"])
            ins = model_kwargs["hidden_size"]
        else:
            ins = input_size[0]
        n_params += MLP.predict_number_params(input_size=(ins + output_size[0],), output_size=(n_basis,), n_basis=1, **model_kwargs)
        n_params += MLP.predict_number_params(input_size=(ins + n_basis,), output_size=output_size, n_basis=1, **model_kwargs)
        return n_params

    def __init__(self,
                 input_size :tuple[int],
                 output_size :tuple[int],
                 data_type :str,
                 n_basis :int =100,
                 model_type :Union[str, type] ="MLP",
                 model_kwargs :dict =dict(),
                 gradient_accumulation :int =1,
                 cross_entropy: bool = False,
                 ):
        super(AutoEncoder, self).__init__(input_size=input_size, output_size=output_size, data_type=data_type,
                                          n_basis=n_basis, model_type=model_type, model_kwargs=model_kwargs,
                                          gradient_accumulation=gradient_accumulation, cross_entropy=cross_entropy)

        # models and optimizers
        if model_type == "CNN":
            self.conv = CNN(input_size=input_size, output_size=(model_kwargs["hidden_size"],), n_basis=1, n_layers=2, learn_basis_functions=False, hidden_size=model_kwargs["hidden_size"])
            ins = model_kwargs["hidden_size"]
        else:
            ins = input_size[0]
        self.encoder = MLP(input_size=(ins + output_size[0],), output_size=(n_basis,), n_basis=1, learn_basis_functions=False, **model_kwargs)
        self.decoder = MLP(input_size=(ins + n_basis,), output_size=output_size, n_basis=1, learn_basis_functions=False, **model_kwargs)

        params = [*self.encoder.parameters()] + [*self.decoder.parameters()]
        if model_type == "CNN":
            params += [*self.conv.parameters()]
        self.opt = torch.optim.Adam(params, lr=1e-3)

    def predict_from_examples(self,
                              example_xs: torch.tensor,
                              example_ys: torch.tensor,
                              xs: torch.tensor,
                              **kwargs):
        # convert images to vectors
        if self.model_type == "CNN":
            example_xs = self.conv(example_xs)
            xs = self.conv(xs)

        # encoder
        encoder_inputs = torch.cat([example_xs, example_ys], dim=-1)
        latent_representation = self.encoder(encoder_inputs).mean(dim=1, keepdim=True)

        # decoder
        decoder_inputs = torch.cat([xs, latent_representation.repeat(1, xs.shape[1], 1)], dim=-1)
        y_hats = self.decoder(decoder_inputs)
        return y_hats
