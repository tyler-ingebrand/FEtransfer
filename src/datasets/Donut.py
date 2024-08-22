from typing import Tuple

import torch
from FunctionEncoder import BaseDataset




class GaussianDonutDataset(BaseDataset):

    def __init__(self, max_radius=0.9, noise=0.1, device:str="auto", **Kwargs):
        super().__init__(input_size=(2,), # this distribution is not conditioned on anything, so we just want to predict the pdf for all two-d inputs
                         output_size=(1,),
                         total_n_functions=float("inf"),
                         total_n_samples_per_function=float("inf"),
                         data_type="stochastic",
                         n_functions_per_sample=10,
                         n_points_per_sample=1_000,
                         device=device,
                         **Kwargs
                         )
        self.max_radius = max_radius
        self.noise = noise
        self.lows = torch.tensor([-1, -1], device=self.device)
        self.highs = torch.tensor([1, 1], device=self.device)
        self.positive_logit = 5
        self.negative_logit = -5
        self.volume = (self.highs - self.lows).prod()
        self.oracle_size = 1


    def sample_info(self):
        radii = torch.rand(self.n_functions_per_sample, device=self.device) * self.max_radius
        oracle_inputs = radii.unsqueeze(1)
        return {"radii": radii, "oracle_inputs": oracle_inputs}

    def sample_inputs(self, info, n_samples):
        radii = info["radii"].unsqueeze(1)

        # generate positive samples
        angles = torch.rand((self.n_functions_per_sample, n_samples // 2), device=self.device) * 2 * torch.pi
        xs = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=-1)
        xs += torch.randn_like(xs, device=self.device) * self.noise

        # generate negative samples
        xs2 = self.sample_negative_inputs(n_functions=self.n_functions_per_sample, n_points=n_samples//2)
        xs = torch.cat([xs, xs2], dim=1)
        return xs

    def compute_outputs(self, xs, info):
        # give high logit to sampled points, and low logit to others
        ys = torch.zeros((self.n_functions_per_sample, xs.shape[1], 1), device=self.device)
        ys[:, :xs.shape[1]//2] = self.positive_logit # first half are positive examples.
        ys[:, xs.shape[1]//2:] = self.negative_logit # second half are negative examples.
        return ys
    def sample(self) -> Tuple[  torch.tensor,
                                torch.tensor,
                                torch.tensor,
                                torch.tensor,
                                dict]:
        # sample functions
        info = self.sample_info()

        # sample inputs
        example_xs = self.sample_inputs(info, n_samples=self.n_examples_per_sample)
        xs = self.sample_inputs(info, n_samples=self.n_points_per_sample)

        # compute outputs
        example_ys = self.compute_outputs(example_xs, info)
        ys = self.compute_outputs(xs, info)

        return example_xs, example_ys, xs, ys, info

    def sample_negative_inputs(self, n_functions, n_points):
        inputs = torch.rand(n_functions, n_points, 2, device=self.device) * (self.highs - self.lows) + self.lows
        return inputs


def get_donut_datasets(device, n_examples):
    type1 = GaussianDonutDataset(device=device,
                                 n_examples_per_sample=n_examples)
    type2 = GaussianDonutDataset(device=device,
                                 n_examples_per_sample=n_examples)
    type3 = GaussianDonutDataset(device=device,
                                 n_examples_per_sample=n_examples)
    type4 = GaussianDonutDataset(device=device,
                                 n_examples_per_sample=n_examples)

    return type1, type2, type3, type4