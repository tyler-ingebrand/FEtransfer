import torch
from typing import Union
from torch import Tensor
from FunctionEncoder import BaseCallback, BaseDataset
from tqdm import trange

class Transformer(torch.nn.Module):
    def __init__(self,
                 input_size :tuple[int],
                 output_size :tuple[int],
                 data_type :str,
                 n_basis :int =100,
                 model_type :Union[str, type] ="MLP",
                 model_kwargs :dict =dict(),
                 gradient_accumulation :int = 1,
                 ):
        super().__init__()
        hidden_size = model_kwargs["hidden_size"]
        nheads = model_kwargs["n_heads"]
        self.transformer = torch.nn.Transformer(d_model=n_basis,
                                                nhead=nheads,
                                                num_encoder_layers=4,
                                                num_decoder_layers=4,
                                                dim_feedforward=hidden_size,
                                                dropout=0.1,
                                                batch_first=True)
        self.encoder_examples = torch.nn.Sequential(
            torch.nn.Linear(input_size[0] + output_size[0], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_basis),
        )
        self.encoder_prediction = torch.nn.Sequential(
            torch.nn.Linear(input_size[0], hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, n_basis),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(n_basis, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size[0]),
        )
        self.opt = torch.optim.Adam([ *self.transformer.parameters(),
                                            *self.encoder_examples.parameters(),
                                            *self.encoder_prediction.parameters(),
                                            *self.decoder.parameters()], lr=1e-3)
        self.data_type = data_type
        self.gradient_accumulation = gradient_accumulation


    def predict_from_examples(self,
                example_xs: torch.tensor, # F x B1 x N size
                example_ys: torch.tensor, # F x B1 x M size
                xs: torch.tensor, # F X B2 x N size
                **kwargs) -> Tensor:        
        examples = torch.cat((example_xs, example_ys), dim=2) # F x B1 x (N + M) size
        example_encodings = self.encoder_examples(examples) # F x B1 x D size
        input_encodings = self.encoder_prediction(xs) # F x B1 x D size

        # forward pass
        output_embedding = self.transformer(example_encodings, input_encodings)
        output = self.decoder(output_embedding)
        return output

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
            # fetch data
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