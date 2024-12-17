import torch
from typing import Union

from FunctionEncoder.Model.Architecture.CNN import CNN
from FunctionEncoder.Model.Architecture.MLP import MLP
from torch import Tensor
from FunctionEncoder import BaseCallback, BaseDataset
from tqdm import trange

from src.algs.BaseAlg import BaseAlg
from src.algs.generic_function_space_methods import _distance

def predict_number_params_transformer_decoder(n_basis, hidden_size):
    num_trans_layers = 4
    kvq_size = 3

    # transformer size - decoder only model
    # size of each layer
    linear1_size = n_basis * hidden_size + hidden_size
    linear2_size = hidden_size * n_basis + n_basis
    norm1_size = 2 * n_basis
    norm2_size = 2 * n_basis
    self_attention_size = (kvq_size * n_basis +
                           kvq_size * n_basis * n_basis +
                           n_basis * n_basis + n_basis)
    layer_size = linear1_size + linear2_size + norm1_size + norm2_size + self_attention_size
    transformer_decoder_size = num_trans_layers * layer_size
    return transformer_decoder_size

class TransformerFunctionalEncoding(BaseAlg):

    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs):
        num_params = 0

        # maybe a cnn
        if model_type == "CNN":
            num_params += CNN.predict_number_params(input_size=input_size, output_size=(model_kwargs["hidden_size"],),  learn_basis_functions=False, n_basis=1, n_layers=4, hidden_size=model_kwargs["hidden_size"])
            input_size = (model_kwargs["hidden_size"], ) # input size to the next layer

        # encoder examples
        num_params += MLP.predict_number_params((input_size[0] + output_size[0],), (n_basis,), learn_basis_functions=False,  n_basis=1, hidden_size=model_kwargs["hidden_size"], n_layers=2)

        # basis functions
        num_params += MLP.predict_number_params(input_size, output_size, n_basis=n_basis, hidden_size=model_kwargs["hidden_size"], n_layers=model_kwargs["n_layers"])

        # decoder
        num_params += MLP.predict_number_params((n_basis,), (n_basis,), n_basis=1, learn_basis_functions=False,  hidden_size=model_kwargs["hidden_size"], n_layers=2)

        # transformer size - decoder only model
        num_params += predict_number_params_transformer_decoder(n_basis, model_kwargs["hidden_size"])

        return num_params


    def __init__(self,
                 input_size :tuple[int],
                 output_size :tuple[int],
                 data_type :str,
                 n_basis :int =100,
                 model_type :Union[str, type] ="MLP",
                 model_kwargs :dict =dict(),
                 gradient_accumulation :int = 1,
                 cross_entropy: bool = False,
                 ):

        super(TransformerFunctionalEncoding, self).__init__(input_size=input_size, output_size=output_size, data_type=data_type,
                                          n_basis=1, model_type=model_type, model_kwargs=model_kwargs,
                                          gradient_accumulation=gradient_accumulation, cross_entropy=cross_entropy)
        hidden_size = model_kwargs["hidden_size"]
        nheads = model_kwargs["n_heads"]

        # converts images to learned vectors first
        if model_type == "CNN":
            self.conv = CNN(input_size=input_size, output_size=(model_kwargs["hidden_size"],), n_basis=1, learn_basis_functions=False, n_layers=4, hidden_size=model_kwargs["hidden_size"])
            input_size = (model_kwargs["hidden_size"], )

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=n_basis, nhead=nheads, dim_feedforward=hidden_size, batch_first=True, layer_norm_eps=1e-3)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.encoder_examples = MLP((input_size[0] + output_size[0],), (n_basis,), n_basis=1,  learn_basis_functions=False, hidden_size=hidden_size, n_layers=2)
        self.basis_functions = MLP(input_size=input_size, output_size=output_size, n_basis=n_basis, hidden_size=hidden_size, n_layers=model_kwargs["n_layers"])
        self.decoder = MLP((n_basis,), (n_basis,), n_basis=1,  learn_basis_functions=False, hidden_size=hidden_size, n_layers=2)
        self.opt = torch.optim.AdamW([ *self.transformer.parameters(),
                                            *self.encoder_examples.parameters(),
                                            *self.basis_functions.parameters(),
                                            *self.decoder.parameters()], lr=1e-3, weight_decay=1e-2)

    def predict_from_examples(self,
                example_xs: torch.tensor, # F x B1 x N size
                example_ys: torch.tensor, # F x B1 x M size
                xs: torch.tensor, # F X B2 x N size
                **kwargs) -> Tensor:        

        # convert images to vectors
        if self.model_type == "CNN":
            example_xs = self.conv(example_xs)
            xs = self.conv(xs)

        # convert all data to encodings
        examples = torch.cat((example_xs, example_ys), dim=2) # F x B1 x (N + M) size
        example_encodings = self.encoder_examples(examples) # F x B1 x D size

        # forward pass
        output_embedding = self.transformer(example_encodings)
        output_coefficients = self.decoder(output_embedding)[:, -1, :]
        basis_functions = self.basis_functions(xs)
        output = torch.einsum("fdmk,fk->fdm", basis_functions, output_coefficients)
        return output

