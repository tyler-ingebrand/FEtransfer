import torch
from typing import Union

from FunctionEncoder.Model.Architecture.CNN import CNN
from FunctionEncoder.Model.Architecture.MLP import MLP
from torch import Tensor
from FunctionEncoder import BaseCallback, BaseDataset
from tqdm import trange

from src.algs.BaseAlg import BaseAlg
from src.algs.generic_function_space_methods import _distance

def predict_number_params_transformer(n_basis, hidden_size):
    # transformer size
    num_trans_layers = 4
    kvq_size = 3

    # first norm layer
    norm1 = n_basis * 2
    # size of each layer
    multihead_attention_size = (kvq_size * n_basis +
                                kvq_size * n_basis * n_basis +
                                n_basis * n_basis + n_basis)
    linear1_size = n_basis * hidden_size + hidden_size
    linear2_size = hidden_size * n_basis + n_basis
    norm1_size = 2 * n_basis
    norm2_size = 2 * n_basis

    # full encoder layer size
    encoder_layer_size = multihead_attention_size + linear1_size + linear2_size + norm1_size + norm2_size
    transformer_encoder_size = num_trans_layers * encoder_layer_size + norm1

    # decoder side of transformer
    norm1 = n_basis * 2
    # size of each layer
    multihead_attention_size = (kvq_size * n_basis +
                                kvq_size * n_basis * n_basis +
                                n_basis * n_basis + n_basis)
    linear1_size = n_basis * hidden_size + hidden_size
    linear2_size = hidden_size * n_basis + n_basis
    norm1_size = 2 * n_basis
    norm2_size = 2 * n_basis
    norm3_size = 2 * n_basis
    self_attention_size = (kvq_size * n_basis +
                           kvq_size * n_basis * n_basis +
                           n_basis * n_basis + n_basis)
    layer_size = multihead_attention_size + linear1_size + linear2_size + norm1_size + norm2_size + norm3_size + self_attention_size
    transformer_decoder_size = num_trans_layers * layer_size + norm1
    return transformer_encoder_size + transformer_decoder_size



class Transformer(BaseAlg):

    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs):
        num_params = 0

        if model_type == "CNN":
            num_params += CNN.predict_number_params(input_size=input_size, output_size=(model_kwargs["hidden_size"],), n_basis=1, learn_basis_functions=False,  n_layers=4, hidden_size=model_kwargs["hidden_size"])
            input_size = (model_kwargs["hidden_size"],)

        # encoder examples
        num_params += MLP.predict_number_params((input_size[0] + output_size[0],), (n_basis,), n_basis=1,  learn_basis_functions=False, hidden_size=model_kwargs["hidden_size"], n_layers=2)

        # encoder prediction
        num_params += MLP.predict_number_params((input_size[0],), (n_basis,), n_basis=1,  learn_basis_functions=False, hidden_size=model_kwargs["hidden_size"], n_layers=2)

        # transformer
        num_params += predict_number_params_transformer(n_basis, model_kwargs["hidden_size"])

        # decoder
        num_params += MLP.predict_number_params((n_basis,), (output_size[0],), n_basis=1, learn_basis_functions=False,  hidden_size=model_kwargs["hidden_size"], n_layers=2)
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

        super(Transformer, self).__init__(input_size=input_size, output_size=output_size, data_type=data_type,
                                          n_basis=1, model_type=model_type, model_kwargs=model_kwargs,
                                          gradient_accumulation=gradient_accumulation, cross_entropy=cross_entropy)
        hidden_size = model_kwargs["hidden_size"]
        nheads = model_kwargs["n_heads"]

        # conv net if needed
        if model_type == "CNN":
            self.conv = CNN(input_size=input_size, output_size=(model_kwargs["hidden_size"],), n_basis=1, n_layers=4,  learn_basis_functions=False, hidden_size=model_kwargs["hidden_size"])
            input_size = (model_kwargs["hidden_size"],)
        self.transformer = torch.nn.Transformer(d_model=n_basis,
                                                nhead=nheads,
                                                num_encoder_layers=4,
                                                num_decoder_layers=4,
                                                dim_feedforward=hidden_size,
                                                dropout=0.1,
                                                batch_first=True)
        self.encoder_examples = MLP((input_size[0] + output_size[0],), (n_basis,), n_basis=1,  learn_basis_functions=False, hidden_size=hidden_size, n_layers=2)
        self.encoder_prediction = MLP(input_size=(input_size[0],), output_size=(n_basis,), n_basis=1,  learn_basis_functions=False, hidden_size=hidden_size, n_layers=2)
        self.decoder = MLP(input_size=(n_basis,), output_size=(output_size[0],), n_basis=1,  learn_basis_functions=False, hidden_size=hidden_size, n_layers=2)
        self.opt = torch.optim.Adam([ *self.transformer.parameters(),
                                            *self.encoder_examples.parameters(),
                                            *self.encoder_prediction.parameters(),
                                            *self.decoder.parameters()], lr=1e-3)
        self.d_model = n_basis

    def predict_from_examples(self,
                example_xs: torch.tensor, # F x B1 x N size
                example_ys: torch.tensor, # F x B1 x M size
                xs: torch.tensor, # F X B2 x N size
                **kwargs) -> Tensor:        

        # convert images to vectors
        if self.model_type == "CNN":
            example_xs = self.conv(example_xs)
            xs = self.conv(xs)

        # encode inputs
        examples = torch.cat((example_xs, example_ys), dim=2) # F x B1 x (N + M) size
        example_encodings = self.encoder_examples(examples) # F x B1 x D size
        input_encodings = self.encoder_prediction(xs) # F x B1 x D size

        # forward pass
        # we need to reshape. Otherwise, the transformer can attend all of the query points.
        # the query points are designed to be independent, so this would be cheating.
        reshaped_input_encodings = input_encodings.view(-1, 1, input_encodings.shape[-1])
        reshaped_example_encodings = example_encodings.unsqueeze(1).repeat(1, xs.shape[1], 1, 1)
        reshaped_example_encodings = reshaped_example_encodings.view(-1, example_xs.shape[1], self.d_model)

        # pass through transformer
        output_embedding = self.transformer(reshaped_example_encodings, reshaped_input_encodings)
        output_embedding = output_embedding.view(xs.shape[0], xs.shape[1], self.d_model)

        # decode outputs
        output = self.decoder(output_embedding)
        return output
