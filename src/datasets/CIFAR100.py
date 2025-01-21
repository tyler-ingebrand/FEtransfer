from typing import Tuple

import matplotlib
import torch
from FunctionEncoder import CIFARDataset, BaseDataset
from matplotlib import pyplot as plt


class ModifiedCIFAR(CIFARDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(logit_scale=1, *args, **kwargs)
        self.oracle_size = 100

    def sample(self, heldout=False) -> Tuple[  torch.tensor,
                                                torch.tensor,
                                                torch.tensor,
                                                torch.tensor,
                                                dict]:
        example_xs, example_ys, xs, ys, info = super().sample(heldout=heldout)
        positive_class_indicies = info["positive_class_indicies"] # 10
        negative_class_indicies = info["negative_class_indicies"] # 10 x 50
        positive_class_indicies = positive_class_indicies.unsqueeze(1).expand(-1, xs.shape[1]//2)
        class_indicies = torch.cat([positive_class_indicies, negative_class_indicies], dim=1)
        one_hot = torch.nn.functional.one_hot(class_indicies, num_classes=self.oracle_size).float()
        info["oracle_inputs"] = one_hot
        info["classes"] = self.classes
        return example_xs, example_ys, xs, ys, info

    def prototypical_network_fetch_data_for_computing_prototypes(self):
        # returns 100 images from all classes, including heldout classes.
        data = self.data_tensor
        # data is of size 100x500xwhc. Need to select 100 random images from every class
        perms = [torch.randperm(data.shape[1], device=self.device)[:100] for _ in range(data.shape[0])]
        perms = torch.stack(perms, dim=0)
        data = data[torch.arange(data.shape[0], device=self.device).unsqueeze(1), perms]

        # also fetch which indicies are for training and testing
        training_class_indicies = self.training_indicies
        testing_class_indicies = self.heldout_indicies
        return data, training_class_indicies, testing_class_indicies




def get_cifar_datasets(device, n_examples, n_functions):
    train = ModifiedCIFAR(device=device, n_examples=n_examples, split="train", n_functions=n_functions)
    type1 = ModifiedCIFAR(device=device, n_examples=n_examples, split="test", n_functions=n_functions)
    type2 = None # no linear combinations of distributions
    type3 = ModifiedCIFAR(device=device, n_examples=n_examples, split="test", heldout_classes_only=True, n_functions=n_functions)
    return train, type1, type2, type3

def plot_cifar(xs, ys, y_hats, example_xs, example_ys, save_dir, type_i, info):
    fig, ax = plt.subplots(4, 12, figsize=(18, 8),
                           gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.2, 1, 1, 1, 1, 0.2, 1, 1]})
    for row in range(min(4, xs.shape[0])):
        # positive examples
        for col in range(4):
            ax[row, col].axis("off")
            img = example_xs[row, col].permute(2, 1, 0).cpu().numpy()
            ax[row, col].imshow(img)
            class_idx = info["positive_example_class_indicies"][row]
            class_name = info["classes"][class_idx]
            ax[row, col].set_title(class_name)

        # negative examples
        for col in range(5, 9):
            ax[row, col].axis("off")
            img = example_xs[row, -col + 4].permute(2, 1, 0).cpu().numpy()
            ax[row, col].imshow(img)
            class_idx = info["negative_example_class_indicies"][row, -col + 4]
            class_name = info["classes"][class_idx]
            ax[row, col].set_title(class_name)

        # disable axis for the two unfilled plots
        ax[row, 4].axis("off")
        ax[row, 9].axis("off")

        # new image and prediction
        ax[row, 10].axis("off")
        img = xs[row, 0].permute(2, 1, 0).cpu().numpy()
        ax[row, 10].imshow(img)

        logits = y_hats[row, 0]
        probs = torch.softmax(logits, dim=-1)
        ax[row, 10].set_title(f"$P(x \in C) = {probs[0].item() * 100:.0f}\%$")

        # add new negative image and prediction
        ax[row, 11].axis("off")
        img = xs[row, -1].permute(2, 1, 0).cpu().numpy()
        ax[row, 11].imshow(img)

        logits = y_hats[row, -1]
        probs = torch.softmax(logits, dim=-1)
        ax[row, 11].set_title(f"$P(x \in C) = {probs[0].item() * 100:.0f}\%$")

    # add dashed lines between positive and negative examples
    left = ax[0, 3].get_position().xmax
    right = ax[0, 5].get_position().xmin
    xpos = (left + right) / 2
    top = ax[0, 3].get_position().ymax + 0.05
    bottom = ax[3, 3].get_position().ymin
    line1 = matplotlib.lines.Line2D((xpos, xpos), (bottom, top), transform=fig.transFigure, color="black",linestyle="--")

    # add dashed lines between negative examples and new image
    left = ax[0, 8].get_position().xmax
    right = ax[0, 10].get_position().xmin
    xpos = (left + right) / 2
    line2 = matplotlib.lines.Line2D((xpos, xpos), (bottom, top), transform=fig.transFigure, color="black",linestyle="--")

    fig.lines = line1, line2,

    # add one text above positive samples
    left = ax[0, 0].get_position().xmin
    right = ax[0, 3].get_position().xmax
    xpos = (left + right) / 2
    ypos = ax[0, 0].get_position().ymax + 0.08
    fig.text(xpos, ypos, "Positive Examples", ha="center", va="center", fontsize=16, weight="bold")

    # add one text above negative samples
    left = ax[0, 5].get_position().xmin
    right = ax[0, 8].get_position().xmax
    xpos = (left + right) / 2
    fig.text(xpos, ypos, "Negative Examples", ha="center", va="center", fontsize=16, weight="bold")

    # add one text above new image
    left = ax[0, 10].get_position().xmin
    right = ax[0, 11].get_position().xmax
    xpos = (left + right) / 2
    fig.text(xpos, ypos, "Query Image", ha="center", va="center", fontsize=16, weight="bold")

    plt.savefig(f"{save_dir}/type{type_i+1}.png")
    plt.clf()


if __name__ == "__main__":
    d = ModifiedCIFAR(device="cpu", n_examples=10, split="train")
