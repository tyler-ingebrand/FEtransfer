import os
from typing import Union

import numpy as np
import torch
from FunctionEncoder import BaseCallback, BaseDataset
from FunctionEncoder.Model.Architecture.CNN import CNN, ConvLayers
from FunctionEncoder.Model.Architecture.MLP import MLP, get_activation
from tensorboard.backend.event_processing import event_accumulator
from tqdm import trange

try:
    from src.algs.BaseAlg import BaseAlg
    from src.algs.generic_function_space_methods import _distance
except: # this is only used if you run the main function
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.algs.BaseAlg import BaseAlg
    from src.algs.generic_function_space_methods import _distance


# These are empirically good values for MAML. Unfortunately, this means we had to tune
# MAML, in contrast to all other algorithms.
# also, its not consistent in the sense that the best learning rate for type 1 transfer != the best learning rate for type 3 transfer
MAML_INTERNAL_LEARNING_RATE = {
    "MAML1":
       {
           "PolynomialDataset": 0.0005,
           "ModifiedCIFAR":  0.0001,
           "SevenScenesDataset": 1e-3,
            "MujoCoAntDataset": 0.005,
       },
    "MAML5":
        {
            "PolynomialDataset": 1e-05,
            "ModifiedCIFAR": 0.0005,
            "SevenScenesDataset": 1e-4,
            "MujoCoAntDataset": 0.0005,
        },
}



class MAML(BaseAlg):

    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs):
        if model_type == "MLP":
            n_params = MLP.predict_number_params(input_size, output_size, n_basis=1,  learn_basis_functions=False, **model_kwargs)
        else:
            n_params = CNN.predict_number_params(input_size, output_size, n_basis=1, learn_basis_functions=False,  **model_kwargs)
        return n_params

    @staticmethod
    def get_layers(model) -> list:
        if isinstance(model, torch.nn.Sequential):
            sublayers = []
            for layer in model:
                sublayers.extend(MAML.get_layers(layer))
        elif isinstance(model, MLP):
            sublayers = MAML.get_layers(model.model)
        elif isinstance(model, CNN):
            sublayers = MAML.get_layers(model.model)
        elif isinstance(model, ConvLayers):
            sublayers = MAML.get_layers(model.model)
        else:
            sublayers = [model]
        return sublayers

    def __init__(self,
                 input_size :tuple[int],
                 output_size :tuple[int],
                 data_type :str,
                 model_type :Union[str, type],
                 model_kwargs :dict =dict(),
                 n_maml_update_steps :int =1,
                 gradient_accumulation :int =1,
                 cross_entropy: bool = False,
                 internal_learning_rate=1e-2,
                 ):
        self.n_maml_update_steps = n_maml_update_steps
        super(MAML, self).__init__(input_size=input_size, output_size=output_size, data_type=data_type,
                                  n_basis=1, model_type=model_type, model_kwargs=model_kwargs,
                                  gradient_accumulation=gradient_accumulation, cross_entropy=cross_entropy)
        self.internal_learning_rate = internal_learning_rate

        # create model and opt
        if model_type == "MLP":
            self.model = MLP(input_size, output_size, n_basis=1,  learn_basis_functions=False, **model_kwargs)
        else:
            self.model = CNN(input_size, output_size, n_basis=1,  learn_basis_functions=False, **model_kwargs)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # vmaps
        func = lambda x, W, b, stride, padding: torch.nn.functional.conv2d(x, W, bias=b, stride=stride, padding=padding)
        self.conv2d_vmap = torch.vmap(func, in_dims=(0,0, 0, None, None))
        func = lambda x, kernel_size, stride: torch.nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride)
        self.maxpool_vmap = torch.vmap(func, in_dims=(0, None, None))
        func = lambda x: x.reshape(x.shape[0], -1)
        self.flatten_vmap = torch.vmap(func)


    def forward_pass_copied_model(self, xs, copied_parameters):
        # copied_parameters is a list of tuples, where each tuple contains the weights and biases for a linear layer
        # xs is the input data
        linear_layer_index = 0
        layers = MAML.get_layers(self.model)

        for layer in layers:
            if isinstance(layer, torch.nn.Linear):
                W_copy, b_copy = copied_parameters[linear_layer_index]
                xs = torch.einsum("fmn,fdn->fdm", W_copy, xs) + b_copy.unsqueeze(1)
                linear_layer_index += 1
            elif isinstance(layer, torch.nn.Conv2d):
                W_copy, b_copy = copied_parameters[linear_layer_index]
                xs = self.conv2d_vmap(xs, W_copy, b_copy, layer.stride, layer.padding)
                linear_layer_index += 1
            elif isinstance(layer, torch.nn.MaxPool2d):
                xs = self.maxpool_vmap(xs, layer.kernel_size, layer.stride)
            elif isinstance(layer, torch.nn.Flatten):
                xs = self.flatten_vmap(xs)
            else:
                xs = layer(xs)
        return xs


    def copy_params(self, num_copies):
        # first generate models based on the current parameters
        # this generates a new model for each example
        params = []
        layers = MAML.get_layers(self.model)
        for layer in layers:
            if isinstance(layer, torch.nn.Linear):
                W_copy = layer.weight.unsqueeze(0).expand(num_copies, -1, -1).clone()
                b_copy = layer.bias.unsqueeze(0).expand(num_copies, -1,).clone()
                params.append((W_copy, b_copy))
            elif isinstance(layer, torch.nn.Conv2d):
                W_copy = layer.weight.unsqueeze(0).expand(num_copies, -1, -1, -1, -1).clone()
                b_copy = layer.bias.unsqueeze(0).expand(num_copies, -1,).clone()
                params.append((W_copy, b_copy))

        return params
    def predict_from_examples(self,
                              example_xs: torch.tensor,
                              example_ys: torch.tensor,
                              xs: torch.tensor, # f x d x n
                              **kwargs):

        # first create a copy of the model parameters
        # we will train this copy
        params = self.copy_params(example_xs.shape[0])

        # next, we will update the models based on the examples via gradient descent for some number of gradient steps
        learning_rate = self.internal_learning_rate
        for _ in range(self.n_maml_update_steps):
            # compute loss
            y_example_hats = self.forward_pass_copied_model(example_xs, params)

            if not self.cross_entropy or self.data_type != "categorical":
                loss = _distance(y_example_hats, example_ys, data_type=self.data_type, squared=True).mean()
            else:
                classes = example_ys.argmax(dim=2)
                loss = torch.nn.CrossEntropyLoss()(y_example_hats.reshape(-1, 2), classes.reshape(-1))


            # back prop, retain graph so we can compute the gradient of the loss w.r.t. the parameters
            grads = torch.autograd.grad(loss, [p for p2 in params for p in p2], create_graph=True)

            # update the parameters by hand, since torch  is not meant for this.
            for i in range(len(params)):
                W, b = params[i]
                W_grad, b_grad = grads[2*i], grads[2*i + 1]
                if torch.isnan(W_grad).any():
                    print("NaN detected")
                if torch.isnan(b_grad).any():
                    print("NaN detected")
                W = W - learning_rate * W_grad
                b = b - learning_rate * b_grad
                params[i] = (W, b)


        # finally, we will predict the output for the new examples
        y_hats = self.forward_pass_copied_model(xs, params)
        if torch.isnan(y_hats).any():
            print("NaN detected")
        return y_hats

