import os
import argparse
import os
from contextlib import nullcontext
from typing import Union

from datetime import datetime
import torch
from FunctionEncoder import *
import time

import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# set font size and type
from names import *

plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams.update({'font.family': FONT})

from src.CustomCallback import CustomCallback
from src.algs.PrototypicalNetwork import ProtoTypicalNetwork
from src.algs.get_model import get_model, predict_number_params, get_number_params, get_hidden_size
from src.datasets.get_dataset import get_datasets, get_plotting_function
from src.algs.Oracle import Oracle
from src.algs.Transformer import get_gradient_accumulation_steps






# script params
fe_100_dir = "logs/experiment/Polynomial/LS/2024-11-26_12-24-51"
fe_3_dir = "logs_3_basis/Polynomial/LS/2025-01-02_13-10-49"
ae_dir = "logs/experiment/Polynomial/AE/2024-11-26_13-42-38"
trans_dir = "logs/experiment/Polynomial/Transformer/2024-12-18_10-48-50"
device = "cuda"
assert os.path.exists(fe_100_dir)
assert os.path.exists(fe_3_dir)
assert os.path.exists(ae_dir)
assert os.path.exists(trans_dir)


# load dataset
train_dataset, type1_dataset, type2_dataset, type3_dataset = get_datasets("Polynomial", device, 200, n_functions=10)

# load models
# get model for fe with 100 basis functions
hidden_size = get_hidden_size("LS", type1_dataset, 1e6, 100, 6, 5) # calculated to have approximately args.n_params
fe_100_model = get_model("LS", type1_dataset, 100, 6, 5, hidden_size, maml_steps=1, cross_entropy=False, maml_internal_learning_rate=1, device=device, gradient_accumulation=1)

# get model for fe with 3 basis functions
hidden_size = get_hidden_size("LS", type1_dataset, 1e6, 3, 6, 5) # calculated to have approximately args.n_params
fe_3_model = get_model("LS", type1_dataset, 3, 6, 5, hidden_size, maml_steps=1, cross_entropy=False, maml_internal_learning_rate=1, device=device, gradient_accumulation=1)

# get model for ae
hidden_size = get_hidden_size("AE", type1_dataset, 1e6, 100, 6, 5) # calculated to have approximately args.n_params
ae_model = get_model("AE", type1_dataset, 100, 6, 5, hidden_size, maml_steps=1, cross_entropy=False, maml_internal_learning_rate=1, device=device, gradient_accumulation=1)

# get model for transformer
hidden_size = get_hidden_size("Transformer", type1_dataset, 1e6, 100, 6, 5) # calculated to have approximately
trans_model = get_model("Transformer", type1_dataset, 100, 6, 5, hidden_size, maml_steps=1, cross_entropy=False, maml_internal_learning_rate=1, device=device, gradient_accumulation=1)

# load models
fe_100_model.load_state_dict(torch.load(os.path.join(fe_100_dir, "model.pth"), weights_only=True))
fe_3_model.load_state_dict(torch.load(os.path.join(fe_3_dir, "model.pth"), weights_only=True))
ae_model.load_state_dict(torch.load(os.path.join(ae_dir, "model.pth"), weights_only=True))
trans_model.load_state_dict(torch.load(os.path.join(trans_dir, "model.pth"), weights_only=True))

# create 4 side by side plots
fig, axs = plt.subplots(1, 3, figsize=(24, 6))
for i, dataset in enumerate([type1_dataset, type2_dataset, type3_dataset]):
    # get data
    example_xs, example_ys, xs, ys, info = dataset.sample()

    # approximate
    with torch.no_grad():
        fe_3_y_hats = fe_3_model.predict_from_examples(example_xs, example_ys, xs, method="least_squares")
        fe_100_y_hats = fe_100_model.predict_from_examples(example_xs, example_ys, xs, method="least_squares")
        ae_y_hats = ae_model.predict_from_examples(example_xs, example_ys, xs)
        trans_y_hats = trans_model.predict_from_examples(example_xs, example_ys, xs)

    # select first dim only
    xs = xs[0, :, 0]
    ys = ys[0, :, 0]
    fe_3_y_hats = fe_3_y_hats[0, :, 0]
    fe_100_y_hats = fe_100_y_hats[0, :, 0]
    ae_y_hats = ae_y_hats[0, :, 0]
    trans_y_hats = trans_y_hats[0, :, 0]


    # sort
    xs, sorted_indices = xs.sort(dim=0)
    ys = ys[sorted_indices]
    fe_3_y_hats = fe_3_y_hats[sorted_indices]
    fe_100_y_hats = fe_100_y_hats[sorted_indices]
    ae_y_hats = ae_y_hats[sorted_indices]
    trans_y_hats = trans_y_hats[sorted_indices]

    # plot  
    axs[i].plot(xs.cpu().numpy(), fe_3_y_hats.cpu().numpy(), label="FE(k=3)", color="tab:blue")
    axs[i].plot(xs.cpu().numpy(), fe_100_y_hats.cpu().numpy(), label="FE(k=100)", color="tab:purple")
    axs[i].plot(xs.cpu().numpy(), ae_y_hats.cpu().numpy(), label="AE", color="tab:red")
    # axs[i].plot(xs.cpu().numpy(), trans_y_hats.cpu().numpy(), label="Transformer", color="tab:red")
    axs[i].plot(xs.cpu().numpy(), ys.cpu().numpy(), label="Ground Truth", color="black", ls="--")

    # labels
    if i == 0:
        axs[i].set_ylabel("f(x)")
    axs[i].set_xlabel("x")
    axs[i].set_title(f"Type {i + 1} Transfer")


# save
plt.legend(fontsize=13)
plt.savefig("plots/canonical.png")
plt.close()


