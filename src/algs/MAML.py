from typing import Union

import torch
from FunctionEncoder import BaseCallback, BaseDataset
from FunctionEncoder.Model.Architecture.MLP import MLP, get_activation
from tqdm import trange

class MAML(torch.nn.Module):
    def __init__(self,
                 input_size :tuple[int],
                 output_size :tuple[int],
                 data_type :str,
                 model_kwargs :dict =dict(),
                 n_maml_update_steps :int =1,
                 gradient_accumulation :int =1,
                 ):

        assert len(input_size) == 1, "Only 1D input supported for now"
        assert input_size[0] >= 1, "Input size must be at least 1"
        assert len(output_size) == 1, "Only 1D output supported for now"
        assert output_size[0] >= 1, "Output size must be at least 1"
        assert data_type in ["deterministic", "stochastic", "categorical"], f"Unknown data type: {data_type}"
        super(MAML, self).__init__()

        # hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.data_type = data_type
        self.n_maml_update_steps = n_maml_update_steps

        # model details
        ins = input_size[0]
        outs = output_size[0]
        hidden_size = model_kwargs["hidden_size"]
        activation = model_kwargs.get("activation", "relu")

        # create model and opt
        layers = []
        layers.append(torch.nn.Linear(ins, hidden_size))
        layers.append(get_activation(activation))
        for _ in range(model_kwargs["n_layers"] - 2):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(get_activation(activation))
        layers.append(torch.nn.Linear(hidden_size, outs))
        self.model = torch.nn.Sequential(*layers)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # accumulates gradients over multiple batches, typically used when n_functions=1 for memory reasons.
        self.gradient_accumulation = gradient_accumulation

        # for printing
        self.model_type = "MLP"
        self.model_kwargs = model_kwargs


    def forward_pass_copied_model(self, xs, copied_parameters):
        # copied_parameters is a list of tuples, where each tuple contains the weights and biases for a linear layer
        # xs is the input data
        linear_layer_index = 0
        for layer in self.model:
            if isinstance(layer, torch.nn.Linear):
                W_copy, b_copy = copied_parameters[linear_layer_index]
                xs = torch.einsum("fmn,fdn->fdm", W_copy, xs) + b_copy.unsqueeze(1)
                linear_layer_index += 1
            else:
                xs = layer(xs)
        return xs


    def predict_from_examples(self,
                              example_xs: torch.tensor,
                              example_ys: torch.tensor,
                              xs: torch.tensor, # f x d x n
                              **kwargs):
        # first generate models based on the current parameters
        # this generates a new model for each example
        params = []
        for layer in self.model:
            if isinstance(layer, torch.nn.Linear):
                W_copy = layer.weight.unsqueeze(0).expand(example_xs.shape[0], -1, -1).clone()
                b_copy = layer.bias.unsqueeze(0).expand(example_xs.shape[0], -1,).clone()
                params.append((W_copy, b_copy))

        # next, we will update the models based on the examples via gradient descent for some number of gradient steps
        learning_rate = 1e-4
        for _ in range(self.n_maml_update_steps):
            # compute loss
            y_example_hats = self.forward_pass_copied_model(example_xs, params)
            loss = self._distance(y_example_hats, example_ys, squared=True).mean()

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
            example_xs, example_ys, xs, ys, _ = dataset.sample()

            # approximate functions, compute error
            y_hats = self.predict_from_examples(example_xs, example_ys, xs)
            prediction_loss = self._distance(y_hats, ys, squared=True).mean()

            # add loss components
            loss = prediction_loss

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

    def _deterministic_inner_product(self,
                                     fs: torch.tensor,
                                     gs: torch.tensor, ) -> torch.tensor:
        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True

        # compute inner products via MC integration
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _stochastic_inner_product(self,
                                  fs: torch.tensor,
                                  gs: torch.tensor, ) -> torch.tensor:
        assert len(fs.shape) in [3, 4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3, 4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2] == 1, f"Expected fs and gs to have the same output size, which is 1 for the stochastic case since it learns the pdf(x), got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True
        assert len(fs.shape) == 4 and len(gs.shape) == 4, "Expected fs and gs to have shape (f,d,m,k)"

        # compute means and subtract them
        mean_f = torch.mean(fs, dim=1, keepdim=True)
        mean_g = torch.mean(gs, dim=1, keepdim=True)
        fs = fs - mean_f
        gs = gs - mean_g

        # compute inner products
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)
        # Technically we should multiply by volume, but we are assuming that the volume is 1 since it is often not known

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _categorical_inner_product(self,
                                   fs: torch.tensor,
                                   gs: torch.tensor, ) -> torch.tensor:
        assert len(fs.shape) in [3, 4], f"Expected fs to have shape (f,d,m) or (f,d,m,k), got {fs.shape}"
        assert len(gs.shape) in [3, 4], f"Expected gs to have shape (f,d,m) or (f,d,m,k), got {gs.shape}"
        assert fs.shape[0] == gs.shape[0], f"Expected fs and gs to have the same number of functions, got {fs.shape[0]} and {gs.shape[0]}"
        assert fs.shape[1] == gs.shape[1], f"Expected fs and gs to have the same number of datapoints, got {fs.shape[1]} and {gs.shape[1]}"
        assert fs.shape[2] == gs.shape[2], f"Expected fs and gs to have the same output size, which is the number of categories in this case, got {fs.shape[2]} and {gs.shape[2]}"

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True
        assert len(fs.shape) == 4 and len(gs.shape) == 4, "Expected fs and gs to have shape (f,d,m,k)"

        # compute means and subtract them
        mean_f = torch.mean(fs, dim=2, keepdim=True)
        mean_g = torch.mean(gs, dim=2, keepdim=True)
        fs = fs - mean_f
        gs = gs - mean_g

        # compute inner products
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _inner_product(self,
                       fs: torch.tensor,
                       gs: torch.tensor) -> torch.tensor:
        if self.data_type == "deterministic":
            return self._deterministic_inner_product(fs, gs)
        elif self.data_type == "stochastic":
            return self._stochastic_inner_product(fs, gs)
        elif self.data_type == "categorical":
            return self._categorical_inner_product(fs, gs)
        else:
            raise ValueError(f"Unknown data type: '{self.data_type}'. Should be 'deterministic', 'stochastic', or 'categorical'")

    def _norm(self, fs: torch.tensor, squared=False) -> torch.tensor:
        norm_squared = self._inner_product(fs, fs)
        if not squared:
            return norm_squared.sqrt()
        else:
            return norm_squared

    def _distance(self, fs: torch.tensor, gs: torch.tensor, squared=False) -> torch.tensor:
        return self._norm(fs - gs, squared=squared)