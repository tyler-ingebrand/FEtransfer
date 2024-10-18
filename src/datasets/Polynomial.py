from typing import Tuple

import torch
from FunctionEncoder import BaseDataset
from matplotlib import pyplot as plt


class PolynomialDataset(BaseDataset):


    def __init__(self,
                 A_range,
                 B_range,
                 C_range,
                 D_range,
                 input_range,
                 **kwargs):
        super().__init__(input_size=(1,),
                         output_size=(1,),
                         total_n_functions=float('inf'),
                         total_n_samples_per_function=float('inf'),
                         n_functions_per_sample=10,
                         n_points_per_sample=1_000,
                         data_type="deterministic",
                         **kwargs)

        self.A_range = torch.tensor(A_range, device=self.device)
        self.B_range = torch.tensor(B_range, device=self.device)
        self.C_range = torch.tensor(C_range, device=self.device)
        self.D_range = torch.tensor(D_range, device=self.device)
        self.input_range = torch.tensor(input_range, device=self.device)
        self.oracle_size = 4

    def sample_info(self):
        As = torch.rand(self.n_functions_per_sample, device=self.device) * (self.A_range[1] - self.A_range[0]) + self.A_range[0]
        Bs = torch.rand(self.n_functions_per_sample, device=self.device) * (self.B_range[1] - self.B_range[0]) + self.B_range[0]
        Cs = torch.rand(self.n_functions_per_sample, device=self.device) * (self.C_range[1] - self.C_range[0]) + self.C_range[0]
        Ds = torch.rand(self.n_functions_per_sample, device=self.device) * (self.D_range[1] - self.D_range[0]) + self.D_range[0]
        oracle_inputs = torch.stack([As, Bs, Cs, Ds], dim=1)
        return {"As": As, "Bs": Bs, "Cs": Cs, "Ds": Ds, "oracle_inputs": oracle_inputs}

    def sample_inputs(self, info, n_samples):
        return torch.rand(info["As"].shape[0], n_samples, self.input_size[0], device=self.device) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

    def compute_outputs(self, inputs, info):
        As = info["As"].reshape(-1, 1, 1)
        Bs = info["Bs"].reshape(-1, 1, 1)
        Cs = info["Cs"].reshape(-1, 1, 1)
        Ds = info["Ds"].reshape(-1, 1, 1)
        return As * inputs ** 3 + Bs * inputs ** 2 + Cs * inputs + Ds

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

def get_polynomial_datasets(device, n_examples):
    train = PolynomialDataset(A_range=(0, 0),
                              B_range=(-3, 3),
                              C_range=(-3, 3),
                              D_range=(-3, 3),
                              input_range=(-5, 5),
                              device=device,
                              n_examples_per_sample=n_examples)
    type1 = PolynomialDataset(A_range=(0, 0),
                              B_range=(-3, 3),
                              C_range=(-3, 3),
                              D_range=(-3, 3),
                              input_range=(-5, 5),
                              device=device,
                              n_examples_per_sample=n_examples)
    type2 = PolynomialDataset(A_range=(0, 0),
                              B_range=(-10, 10),
                              C_range=(-10, 10),
                              D_range=(-10, 10),
                              input_range=(-5, 5),
                              device=device,
                              n_examples_per_sample=n_examples)
    type3 = PolynomialDataset(A_range=(-3, 3),
                              B_range=(-3, 3),
                              C_range=(-3, 3),
                              D_range=(-3, 3),
                              input_range=(-5, 5),
                              device=device,
                              n_examples_per_sample=n_examples)
    return train, type1, type2, type3


def plot_polynomial(xs, ys, y_hats, example_xs, example_ys, save_dir, type_i, info):
    # plot
    n_plots = 9
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))

    # loop
    for i in range(n_plots):
        # sort
        x,y, y_hat = xs[i, :, 0], ys[i, :, 0], y_hats[i, :, 0]
        x, indices = torch.sort(x)
        y = y[indices]
        y_hat = y_hat[indices]

        # plot
        ax = axs[i // 3, i % 3]
        ax.plot(x.cpu(), y.cpu(), label="Groundtruth")
        ax.plot(x.cpu(), y_hat.cpu(), label="Estimate")
        if i == 8:
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/type{type_i+1}.png")
    plt.clf()
