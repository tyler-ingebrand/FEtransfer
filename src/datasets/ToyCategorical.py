from typing import Tuple

import matplotlib
import torch
from FunctionEncoder import CIFARDataset, BaseDataset, CategoricalDataset
from matplotlib import pyplot as plt


class ToyCategorical(BaseDataset):

    def __init__(self,
                 unseen_distributions=False,
                 input_range=(0, 1),
                 n_functions_per_sample: int = 10,
                 n_examples_per_sample: int = 200,
                 n_points_per_sample: int = 1_000,
                 logit_scale=1,
                 device: str = "auto",
                 ):
        super().__init__(input_size=(1,),
                         output_size=(2,),
                         total_n_functions=float('inf'),
                         total_n_samples_per_function=float('inf'),
                         data_type="categorical",
                         n_functions_per_sample=n_functions_per_sample,
                         n_examples_per_sample=n_examples_per_sample,
                         n_points_per_sample=n_points_per_sample,
                         device=device,
                         )
        self.n_categories = 2
        self.input_range = torch.tensor(input_range, device=self.device)
        self.logit_scale = logit_scale
        self.unseen_distributions = unseen_distributions
        self.oracle_size = 8 # OHE(category1), boundary location, OHE(category2), boundary location, OHE(category3)
                             # 2 dims,       , 1 dim,            2 dims,       , 1 dim,            2 dims,
    def states_to_logits(self, xs: torch.tensor, categories: torch.tensor, boundaries: torch.tensor, n_functions, n_examples, ) -> torch.tensor:
        indexes = torch.stack([torch.searchsorted(b, x) for b, x in zip(boundaries, xs)])  # this is the index in the boundary list, need to convert it to index in the category list
        chosen_categories = torch.stack([c[i] for c, i in zip(categories, indexes)])
        logits = torch.zeros(n_functions, n_examples, self.n_categories, device=self.device)
        logits = logits.scatter(2, chosen_categories, 1)
        logits *= self.logit_scale
        return logits

    def sample(self) -> Tuple[  torch.tensor,
                                torch.tensor,
                                torch.tensor,
                                torch.tensor,
                                dict]:
        with torch.no_grad():
            n_functions = self.n_functions_per_sample
            n_examples = self.n_examples_per_sample
            n_points = self.n_points_per_sample

            # generate n_functions sets of coefficients
            # each coefficient is a boundary condition
            # the training set only splits the input space into two intervals, each of which gets a category
            # e.g. x \in [0, 0.5) -> category 0, x \in [0.5, 1) -> category 1
            # the unseen set splits the input space into three intervals, each of which gets a category
            # e.g. x \in [0, 0.33) -> category 0, x \in [0.33, 0.66) -> category 1, x \in [0.66, 1) -> category 0
            n_boundaries = self.n_categories
            boundaries = torch.rand((n_functions, n_boundaries), device=self.device) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
            if not self.unseen_distributions:
                boundaries[:, 0] = self.input_range[1]
            boundaries = torch.sort(boundaries, dim=1).values

            # generate labels, each segment becomes a category
            categories = torch.stack([torch.randperm(self.n_categories, device=self.device) for _ in range(n_functions)])

            # the last category is the same as the first, so append it
            categories = torch.cat([categories, categories[:, 0:1]], dim=1)

            # now generate input data
            example_xs = torch.rand(n_functions, n_examples, 1, device=self.device) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]
            xs = torch.rand(n_functions, n_points, 1, device=self.device) * (self.input_range[1] - self.input_range[0]) + self.input_range[0]

            # generate labels with high logits for the correct category and low logits for the others
            example_logits = self.states_to_logits(example_xs, categories, boundaries, n_functions, n_examples)
            logits = self.states_to_logits(xs, categories, boundaries, n_functions, n_points)

            # create info dict
            oracle_info = [torch.nn.functional.one_hot(categories[:, 0], num_classes=self.n_categories),
                            boundaries[:, 0:1],
                            torch.nn.functional.one_hot(categories[:, 1], num_classes=self.n_categories),
                            boundaries[:, 1:2],
                            torch.nn.functional.one_hot(categories[:, 2], num_classes=self.n_categories)]
            oracle_info = torch.cat(oracle_info, dim=1)
            info = {"boundaries": boundaries, "categories": categories, "oracle_inputs": oracle_info}

        # the output for the first function should be chosen_categories[0][indexes[0]]
        return example_xs, example_logits, xs, logits, info



def get_toy_categetorical_datasets(device, n_examples):
    train = ToyCategorical(device=device, n_examples_per_sample=n_examples)
    type1 = ToyCategorical(device=device, n_examples_per_sample=n_examples)
    type2 = None # no linear combinations of distributions
    type3 = ToyCategorical(device=device, n_examples_per_sample=n_examples, unseen_distributions=True)
    return train, type1, type2, type3

def plot_toy_categorical(xs, ys, y_hats, example_xs, example_ys, save_dir, type_i, info):

    # sort the data bsed on the x values
    xs, indicies = torch.sort(xs, dim=1)
    expanded_indicies = indicies.expand(-1, -1, 2)
    ys = ys.gather(dim=-2, index=expanded_indicies)
    y_hats = y_hats.gather(dim=-2, index=expanded_indicies)

    # get ground truth
    most_likely_category = y_hats.argmax(dim=2, keepdim=True)
    boundaries = info["boundaries"]
    ground_truth_categories = info["categories"]

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(9):
        ax = axs[i // 3, i % 3]

        # plot dashed lines at the boundaries
        label = "Groundtruth Boundary"
        for b in boundaries[i]:
            ax.axvline(b.item(), color="black", linestyle="--", label=label)
            label = None

        # add text labeling the sections with A, B, C, etc
        boundaries_i = boundaries[i]
        boundaries_i = torch.cat([torch.tensor([0]), boundaries_i.to("cpu"), torch.tensor([1])])
        for j in range(len(boundaries_i) - 1):
            a, b = boundaries_i[j], boundaries_i[j + 1]
            if torch.abs(a - b).item() > 0.05:
                ax.text((a.item() + b.item()) / 2, 0.5, chr(65 + ground_truth_categories[i][j]), fontsize=30,
                        verticalalignment="center", horizontalalignment="center")

        # plot predictions
        ax.plot(xs[i].cpu(), most_likely_category[i].cpu(), label="Predicted Category")

        # legend
        if i == 8:
            ax.legend()

        # change yaxis labels to A, B, C, etc
        ax.set_yticks(range(2))
        ax.set_yticklabels(["A", "B"])

        ax.set_xlim(0, 1)
        # y_min, y_max = ys[i].min().item(), ys[i].max().item()
        # ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/type{type_i+1}.png")
    plt.clf()
