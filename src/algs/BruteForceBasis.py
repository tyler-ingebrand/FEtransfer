from typing import Union

import torch
from FunctionEncoder import BaseCallback, BaseDataset
from FunctionEncoder.Model.Architecture.MLP import MLP, get_activation
from tqdm import trange


class BruteForceBasis(torch.nn.Module):
    def __init__(self,
                 input_size :tuple[int],
                 output_size :tuple[int],
                 data_type :str,
                 num_basis:int,
                 num_data :int,
                 model_kwargs :dict =dict(),
                 gradient_accumulation :int =1,
                 ):

        assert len(input_size) == 1, "Only 1D input supported for now"
        assert input_size[0] >= 1, "Input size must be at least 1"
        assert len(output_size) == 1, "Only 1D output supported for now"
        assert output_size[0] >= 1, "Output size must be at least 1"
        assert data_type in ["deterministic", "stochastic", "categorical"], f"Unknown data type: {data_type}"
        super(BruteForceBasis, self).__init__()

        # hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.n_basis = num_basis
        self.data_type = data_type
        self.num_data = num_data

        # model details
        ins_example = (input_size[0] + output_size[0]) * num_data
        outs_example = num_basis
        ins_basis = input_size[0]
        outs_basis = num_basis * output_size[0]
        hidden_size = model_kwargs["hidden_size"]
        activation = model_kwargs.get("activation", "relu")

        # create example to basis model
        layers = []
        layers.append(torch.nn.Linear(ins_example, hidden_size))
        layers.append(get_activation(activation))
        for _ in range(model_kwargs["n_layers"] - 2):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(get_activation(activation))
        layers.append(torch.nn.Linear(hidden_size, outs_example))
        self.example_to_basis = torch.nn.Sequential(*layers)

        # create basis
        layers = []
        layers.append(torch.nn.Linear(ins_basis, hidden_size))
        layers.append(get_activation(activation))
        for _ in range(model_kwargs["n_layers"] - 2):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(get_activation(activation))
        layers.append(torch.nn.Linear(hidden_size, outs_basis))
        self.basis = torch.nn.Sequential(*layers)

        # opti
        self.opt = torch.optim.Adam([ *self.example_to_basis.parameters(), *self.basis.parameters()], lr=1e-3)

        # accumulates gradients over multiple batches, typically used when n_functions=1 for memory reasons.
        self.gradient_accumulation = gradient_accumulation

        # for printing
        self.model_type = "MLP"
        self.model_kwargs = model_kwargs


    def predict_from_examples(self,
                              example_xs: torch.tensor,
                              example_ys: torch.tensor,
                              xs: torch.tensor, # f x d x n
                              **kwargs):
        # create input as all examples and the inputs to predict
        examples = torch.cat([example_xs, example_ys], dim=-1) # f x e x n+m
        examples = examples.reshape(examples.shape[0], -1) # f x (n+m)*e

        # pass through example_to_basis
        basis_coefficients = self.example_to_basis(examples) # f x k

        # pass through basis
        basis_functions = self.basis(xs) # f x d x k*m
        basis_functions = basis_functions.reshape(basis_functions.shape[0], basis_functions.shape[1], example_ys.shape[-1], self.n_basis) # f x d x m x k


        # compute linear combination
        y_hats = torch.einsum("fdmk,fk->fdm", basis_functions, basis_coefficients)
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