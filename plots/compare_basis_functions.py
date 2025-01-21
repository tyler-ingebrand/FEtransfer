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


from src.algs.get_model import get_model, predict_number_params, get_number_params, get_hidden_size
from src.datasets.get_dataset import get_datasets, get_plotting_function
from src.algs.Oracle import Oracle
from src.algs.Transformer import get_gradient_accumulation_steps

# set font size and type
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams.update({'font.family': FONT})


with torch.no_grad():


    device="cuda"
    multiheaded_dir = "logs/experiment/Polynomial/LS/2024-11-26_12-24-56"
    parallel_dir = "logs/parallel/Polynomial/LS-Parallel/2025-01-06_10-05-49"

    assert os.path.exists(multiheaded_dir)
    assert os.path.exists(parallel_dir)


    # load dataset
    train_dataset, type1_dataset, type2_dataset, type3_dataset = get_datasets("Polynomial", device, 200, n_functions=10)

    # load models
    # get model for fe with 100 basis functions
    hidden_size = get_hidden_size("LS", type1_dataset, 1e6, 100, 6, 5) # calculated to have approximately args.n_params
    multiheaded_model = get_model("LS", type1_dataset, 100, 6, 5, hidden_size, maml_steps=1, cross_entropy=False, maml_internal_learning_rate=1, device=device, gradient_accumulation=1)

    # get model for fe with 3 basis functions
    hidden_size = get_hidden_size("LS-Parallel", type1_dataset, 1e6, 100, 6, 5) # calculated to have approximately args.n_params
    parallel_model = get_model("LS-Parallel", type1_dataset, 100, 6, 5, hidden_size, maml_steps=1, cross_entropy=False, maml_internal_learning_rate=1, device=device, gradient_accumulation=1)

    # load
    multiheaded_model.load_state_dict(torch.load(os.path.join(multiheaded_dir, "model.pth"), weights_only=True))
    parallel_model.load_state_dict(torch.load(os.path.join(parallel_dir, "model.pth"), weights_only=True))

    # create input data and forward pass both
    xs = torch.linspace(-5, 5, 1000).to(device)
    xs = xs.view(-1, 1)
    ys_multiheaded = multiheaded_model.model(xs)
    ys_parallel = parallel_model.model(xs)

    # reshape
    xs = xs[:, 0].cpu().numpy()
    ys_multiheaded = ys_multiheaded[:, 0].cpu().numpy()
    ys_parallel = ys_parallel[:, 0].cpu().numpy()

    # remove most functions
    n_to_plot = 100
    ys_multiheaded = ys_multiheaded[:, :n_to_plot]
    ys_parallel = ys_parallel[:, :n_to_plot]

    # print std devs of all functions in parallel
    # print(f"std devs of all functions in parallel: {ys_parallel.std(axis=0)}")
    # flat_indices = ys_parallel.std(axis=0) < 0.1
    # flat_functions = ys_parallel[:, flat_indices]
    # num_above, num_below = 0, 0
    # for index in range(flat_functions.shape[1]):
    #     if (flat_functions[:, index] > 0.7).all():
    #         num_above += 1
    #     if (flat_functions[:, index] < -0.7).all():
    #         num_below += 1
    # print(f"num above 0.7: {num_above}")
    # print(f"num below -0.7: {num_below}")

    # plot side by side
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(xs, ys_multiheaded)
    axs[0].set_title("Multi-Headed NN")
    axs[0].set_ylabel("f(x)")
    axs[0].set_xlabel("x")

    axs[1].plot(xs, ys_parallel)
    axs[1].set_title("Parallel NNs")
    axs[1].set_xlabel("x")
    
    plt.savefig("compare_basis_functions.png")


