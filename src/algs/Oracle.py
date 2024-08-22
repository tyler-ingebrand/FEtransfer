from typing import Union

import torch
from FunctionEncoder import BaseCallback, BaseDataset
from tqdm import trange


class Oracle(torch.nn.Module):
    def __init__(self,
                 input_size :tuple[int],
                 output_size :tuple[int],
                 oracle_size:int,
                 data_type :str,
                 model_type :Union[str, type] ="MLP",
                 model_kwargs :dict =dict(),
                 gradient_accumulation :int =1,
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

        assert len(input_size) == 1, "Only 1D input supported for now"
        assert input_size[0] >= 1, "Input size must be at least 1"
        assert len(output_size) == 1, "Only 1D output supported for now"
        assert output_size[0] >= 1, "Output size must be at least 1"
        assert data_type in ["deterministic", "stochastic", "categorical"], f"Unknown data type: {data_type}"
        super(Oracle, self).__init__()

        # hyperparameters
        self.input_size = input_size
        self.output_size = output_size
        self.oracle_size = oracle_size
        self.data_type = data_type

        hidden_size = model_kwargs["hidden_size"]
        n_layers = model_kwargs["n_layers"]

        # models and optimizers
        layers = []
        layers.append(torch.nn.Linear(self.input_size[0] + self.oracle_size, hidden_size))
        layers.append(torch.nn.ReLU())
        for _ in range(n_layers - 2):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_size, self.output_size[0]))
        self.model = torch.nn.Sequential(*layers)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # accumulates gradients over multiple batches, typically used when n_functions=1 for memory reasons.
        self.gradient_accumulation = gradient_accumulation

        # for printing
        self.model_type = model_type
        self.model_kwargs = model_kwargs

    def predict_from_examples(self,
                              example_xs: torch.tensor,
                              example_ys: torch.tensor,
                              xs: torch.tensor,
                              info: dict,
                              **kwargs):
        oracle_inputs = info["oracle_inputs"]
        inputs = torch.cat((xs, oracle_inputs.unsqueeze(1).expand(-1, xs.shape[1], -1)), dim=-1)
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
            example_xs, example_ys, xs, ys, _ = dataset.sample()

            # approximate functions, compute error
            y_hats = self.predict_from_examples(example_xs, example_ys, xs, _)
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