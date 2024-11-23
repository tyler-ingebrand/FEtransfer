from typing import Union

import torch
from FunctionEncoder import BaseDataset, BaseCallback
from FunctionEncoder.Model.Architecture.CNN import CNN
from FunctionEncoder.Model.Architecture.MLP import MLP
from tqdm import trange

from src.algs.BaseAlg import BaseAlg
from src.algs.generic_function_space_methods import _distance


class ProtoTypicalNetwork(BaseAlg):

    @staticmethod
    def predict_number_params(input_size, output_size, n_basis, model_type, model_kwargs):
        n_params = 0
        if model_type == "CNN":
            n_params += CNN.predict_number_params(input_size=input_size, output_size=(model_kwargs["hidden_size"],),
                                                  n_basis=1, n_layers=3, hidden_size=model_kwargs["hidden_size"])
            ins = model_kwargs["hidden_size"]
        else:
            ins = input_size[0]
        n_params += MLP.predict_number_params(input_size=(ins,), output_size=(n_basis,), n_basis=1,
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
                 cross_entropy: bool = False, # proto has its own loss functions and does not use cross entropy flag.
                 ):
        super(ProtoTypicalNetwork, self).__init__(input_size=input_size, output_size=output_size, data_type=data_type,
                                          n_basis=n_basis, model_type=model_type, model_kwargs=model_kwargs,
                                          gradient_accumulation=gradient_accumulation, cross_entropy=cross_entropy)

        # models and optimizers
        if model_type == "CNN":
            self.conv = CNN(input_size=input_size, output_size=(model_kwargs["hidden_size"],), n_basis=1, n_layers=3,
                            hidden_size=model_kwargs["hidden_size"])
            ins = model_kwargs["hidden_size"]
        else:
            ins = input_size[0]
        self.network = MLP(input_size=(ins, ), output_size=(n_basis,), n_basis=1, **model_kwargs)
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    def predict_from_examples(self,
                              example_xs: torch.tensor,
                              example_ys: torch.tensor,
                              xs: torch.tensor,
                              info:dict,
                              **kwargs):
        # during training, we only compare against the prototypes of the training classes.
        # during type1 testing, we also only compare against the training classes but on unseen images.
        # during type3 testing, we compare against all classes.
        dataset_type = "train"
        if info["classes_idx"][0] in self.testing_indicies:
            dataset_type = "type3"

        # do forward pass to get probs
        latents = self.network(self.conv(xs))
        distances = ((latents.unsqueeze(1) - self.prototypes.unsqueeze(0).unsqueeze(2))**2).sum(dim=-1)
        distribution = torch.nn.functional.log_softmax(-distances, dim=1)

        # if train/type1, remove unseen classes
        if dataset_type == "train":
            mask = torch.zeros_like(distribution, dtype=torch.bool)
            mask[:, self.testing_indicies] = True
            distribution = torch.where(mask, -float('inf'), distribution)
            # distribution = torch.where(torch.arange(distribution.shape[1], device=distribution.device).unsqueeze(0).unsqueeze(-1) == self.testing_indicies.unsqueeze(1).unsqueeze(2), -float("inf"), distribution)


        # lastly, we need to use the distribution to get the probability that it belongs to the example class, rather than the negative class.
        distribution = torch.softmax(distribution, dim=1)
        correct_classes = info["positive_class_indicies"]
        positive_probability = distribution[torch.arange(distribution.shape[0], device=distribution.device), correct_classes, :]
        negative_probability = torch.sum(distribution, dim=1) - positive_probability
        probs =  torch.stack([positive_probability, negative_probability], dim=-1)

        # convert back to logits
        y_hats = torch.log(probs + 1e-8)
        # make sure maximum difference is 2, since this is what the groundtruth is assumed to be.
        # proto is not optimizing for this distance, but this makes the results more comparable to the others anyway.
        mins = y_hats.min(dim=-1).values.unsqueeze(-1)
        y_hats = y_hats.clamp(min=mins, max=mins+2)
        return y_hats



    def compute_prototypes(self, images, training_class_indicies, testing_class_indicies ):
        latent_images = self.network(self.conv(images))
        self.prototypes = latent_images.mean(dim=1)
        self.training_indicies = training_class_indicies
        self.testing_indicies = testing_class_indicies


    def train_model(self,
                    dataset: BaseDataset,
                    epochs: int,
                    progress_bar=True,
                    callback: BaseCallback = None):
        # Let callbacks few starting data
        if callback is not None:
            # special prototypical method to compute the embeddings for all (training) classes. This is used to compute the prototypes.
            images, training_class_indicies, testing_class_indicies = dataset.prototypical_network_fetch_data_for_computing_prototypes()
            self.compute_prototypes(images, training_class_indicies, testing_class_indicies)
            callback.on_training_start(locals())

        losses = []
        bar = trange(epochs) if progress_bar else range(epochs)
        for epoch in bar:
            # special prototypical method to compute the embeddings for all (training) classes. This is used to compute the prototypes.
            images, training_class_indicies, testing_class_indicies = dataset.prototypical_network_fetch_data_for_computing_prototypes()
            self.compute_prototypes(images, training_class_indicies, testing_class_indicies)

            example_xs, example_ys, xs, ys, _ = dataset.sample()
            info = _
            # we actually need info here because this code was not designed with prototypical networks in mind.

            # get embeddings
            latents = self.network(self.conv(xs))
            # get distribution over training classes
            distances = ((latents.unsqueeze(1) - self.prototypes.unsqueeze(0).unsqueeze(2))**2).sum(dim=-1)
            distribution = torch.nn.functional.log_softmax(-distances, dim=1)
            # of size n_functions x n_classes x n_points

            # set the testing logits to -inf, since they should be unseen effectively.
            # distribution[:, self.testing_indicies] = -float('inf') # this involves inplace op, and is not autogradable
            mask = torch.zeros_like(distribution, dtype=torch.bool)
            mask[:, self.testing_indicies] = True
            distribution = torch.where(mask, -float('inf'), distribution)

            # get the correct classes
            positive_class_indicies = info["positive_class_indicies"].unsqueeze(1).repeat(1, xs.shape[1]//2)
            negative_class_indicies = info["negative_class_indicies"]
            class_indicies = torch.cat([positive_class_indicies, negative_class_indicies], dim=1)
            # of size n_functions x n_points

            # reshape the distribution to be n_functions x n_points x n_classes
            distribution = distribution.permute(0, 2, 1)
            # of size n_functions x n_points x n_classes

            # now squeeze both to be in the proper negative log likelihood format
            distribution = distribution.reshape(-1, distribution.shape[-1])
            class_indicies = class_indicies.reshape(-1)

            # nll
            loss = torch.nn.functional.nll_loss(distribution, class_indicies)

            # TEMP FOR logging
            with torch.no_grad():
                tb = callback[0].tensorboard
                tb.add_scalar("proto_debug/loss", loss, epoch)


            # backprop with gradient clipping
            loss.backward()
            if (epoch + 1) % self.gradient_accumulation == 0:
                norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                self.opt.step()
                self.opt.zero_grad()

            # callbacks

            if callback is not None:
                # special prototypical method to compute the embeddings for all (training) classes. This is used to compute the prototypes.
                images, training_class_indicies, testing_class_indicies = dataset.prototypical_network_fetch_data_for_computing_prototypes()
                self.compute_prototypes(images, training_class_indicies, testing_class_indicies)
                callback.on_step(locals())

        # let callbacks know its done
        if callback is not None:
            # special prototypical method to compute the embeddings for all (training) classes. This is used to compute the prototypes.
            images, training_class_indicies, testing_class_indicies = dataset.prototypical_network_fetch_data_for_computing_prototypes()
            self.compute_prototypes(images, training_class_indicies, testing_class_indicies)
            callback.on_training_end(locals())

