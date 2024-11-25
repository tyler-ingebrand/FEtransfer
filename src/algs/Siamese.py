from typing import Union

import torch
from FunctionEncoder import BaseDataset, BaseCallback
from FunctionEncoder.Model.Architecture.CNN import CNN
from FunctionEncoder.Model.Architecture.MLP import MLP
from tqdm import trange

from src.algs.BaseAlg import BaseAlg
from src.algs.generic_function_space_methods import _distance


class SiameseNetwork(BaseAlg):

    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs):
        n_params = 0
        if model_type == "CNN":
            n_params += CNN.predict_number_params(input_size=input_size, output_size=(model_kwargs["hidden_size"],),
                                                  n_basis=1, n_layers=3,  learn_basis_functions=False, hidden_size=model_kwargs["hidden_size"])
            ins = model_kwargs["hidden_size"]
        else:
            ins = input_size[0]
        n_params += MLP.predict_number_params(input_size=(ins,), output_size=(n_basis,), n_basis=1,
                                              learn_basis_functions=False,
                                              **model_kwargs)
        return n_params

    def __init__(self,
                 input_size: tuple[int],
                 output_size: tuple[int],
                 data_type: str,
                 n_basis: int = 100,
                 model_type: Union[str, type] = "CNN",
                 model_kwargs: dict = dict(),
                 gradient_accumulation: int = 1,
                 cross_entropy: bool = False, # note Siamese has its own loss functions and so ce cannot be used anyway.
                 ):
        super(SiameseNetwork, self).__init__(input_size=input_size, output_size=output_size, data_type=data_type,
                                          n_basis=n_basis, model_type=model_type, model_kwargs=model_kwargs,
                                          gradient_accumulation=gradient_accumulation, cross_entropy=cross_entropy)

        # models and optimizers
        if model_type == "CNN":
            self.conv = CNN(input_size=input_size, output_size=(model_kwargs["hidden_size"],), n_basis=1, n_layers=3,
                            learn_basis_functions=False,
                            hidden_size=model_kwargs["hidden_size"])
            ins = model_kwargs["hidden_size"]
        else:
            ins = input_size[0]
        self.network = MLP(input_size=(ins, ), output_size=(n_basis,), n_basis=1, learn_basis_functions=False,  **model_kwargs)
        self.threshold = 0.1 # this is to prevent the loss function on focusing on sending negative examples to infinite distance apart

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    def predict_from_examples(self,
                              example_xs: torch.tensor,
                              example_ys: torch.tensor,
                              xs: torch.tensor,
                              **kwargs):
        example_latents = self.network(self.conv(example_xs))
        latents = self.network(self.conv(xs))
        pairwise_distances = ((example_latents.unsqueeze(1) - latents.unsqueeze(2))**2).sum(dim=-1)
        min_indicies = torch.argmin(pairwise_distances, dim=2)
        labels = torch.gather(example_ys, 1, min_indicies.unsqueeze(-1).expand(-1, -1, 2))
        return labels


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
            example_xs, example_ys, query_xs, query_ys, _ = dataset.sample()

            # get embeddings
            example_latents = self.network(self.conv(example_xs))
            latents = self.network(self.conv(query_xs))

            # compute pairwise distance
            pairwise_distances = ((example_latents.unsqueeze(1) - latents.unsqueeze(2))**2).sum(dim=-1)

            # compute contrastive loss
            positive_label_distances = pairwise_distances[:, :pairwise_distances.shape[1]//2, :pairwise_distances.shape[2]//2]
            negative_label_distances = pairwise_distances[:, :pairwise_distances.shape[1]//2, pairwise_distances.shape[2]//2:]

            loss = torch.clamp(positive_label_distances - negative_label_distances + self.threshold, min=0)
            loss = loss.mean()

            # TEMP FOR logging
            tb = callback[0].tensorboard
            tb.add_scalar("siamese_debug/positive_distance", positive_label_distances.mean(), epoch)
            tb.add_scalar("siamese_debug/negative_distance", negative_label_distances.mean(), epoch)
            tb.add_scalar("siamese_debug/loss", loss, epoch)

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

