import argparse
import os
from contextlib import nullcontext
from typing import Union

from datetime import datetime
import torch
from FunctionEncoder import *
import time

from src.CustomCallback import CustomCallback
from src.algs.get_model import get_model, predict_number_params, get_number_params, get_hidden_size
from src.datasets.get_dataset import get_datasets, get_plotting_function
from src.algs.Oracle import Oracle


def load_args(args):
    load_dir = args.load_dir
    args = torch.load(f"{load_dir}/args.pth")
    args.load_dir = load_dir
    return args

def check_args(args):
    acceptable_algs = ["FE", "AE", "Transformer", "TFE", "Oracle", "BF", "BFB", "MAML"]
    acceptable_datasets = ["Polynomial", "Donut"]
    assert args.n_basis >=1, f"n_basis must be at least 1, got {args.n_basis}"
    assert args.n_examples >= 1, f"n_examples must be at least 1, got {args.n_examples}"
    assert args.epochs >= 1, f"epochs must be at least 1, got {args.epochs}"
    assert args.algorithm in acceptable_algs, f"algorithm must be in f{acceptable_algs}, got {args.algorithm}"
    assert args.dataset in acceptable_datasets, f"dataset must be in {acceptable_datasets}, got {args.dataset}"
    assert args.n_params >= 1, f"n_params must be at least 1, got {args.n_params}"
    assert args.n_layers >= 1, f"n_layers must be at least 1, got {args.n_layers}"
    assert args.device in ["cpu", "cuda"] or type(args.device) is int, f"device must be in ['cpu', 'cuda'] or an integer, got {args.device}"
    assert args.method in ["least_squares", "inner_product"], f"train_method must be in ['least_squares', 'inner_product'], got {args.method}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--n_basis", type=int, default=100)
    parser.add_argument("--n_examples", type=int, default=200)
    parser.add_argument("--algorithm", type=str, default="FE")
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--dataset", type=str, default="Polynomial")
    parser.add_argument("--n_params", type=int, default=int(1e6))
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--n_heads", type=int, default=5)
    parser.add_argument("--method", type=str, default="least_squares")
    parser.add_argument("--maml_steps", type=int, default=1)
    args = parser.parse_args()


    # find device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if type(args.device) is int:
        args.device = f"cuda:{args.device}"

    # load arguments if they exist
    if args.load_dir is not None:
        args = load_args(args)

    # check arguments for valid settings only.
    check_args(args)

    # set seed
    torch.manual_seed(args.seed)

    # create save directory
    if args.load_dir is None:
        datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        alg_save_name = args.algorithm
        if args.algorithm == "FE":
            alg_save_name += f'_{args.method}'
        elif args.algorithm == "MAML":
            alg_save_name += f'_{args.maml_steps}'
        save_dir = f"{args.log_dir}/{args.dataset}/{alg_save_name}/{datetime}"
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = args.load_dir


    # get datasets
    type1_dataset, type2_dataset, type3_dataset, type4_dataset = get_datasets(args.dataset, args.device, args.n_examples)

    # get model
    args.hidden_size = get_hidden_size(args.algorithm, type1_dataset, args.n_params, args.n_basis, args.n_layers, args.n_heads)
    model = get_model(args.algorithm, type1_dataset, args.n_basis, args.n_layers, args.n_heads, args.hidden_size, method=args.method, maml_steps=args.maml_steps)
    model = model.to(args.device)

    # print("\n\n\n\n")
    # predict_number_params(args.algorithm, args.n_basis, args.n_layers, type1_dataset.input_size, type1_dataset.output_size, args.hidden_size, args.n_heads)

    # assert right size
    model_n_params = get_number_params(model)
    estimated_n_params = predict_number_params(args.algorithm, args.n_basis, args.n_layers, type1_dataset.input_size, type1_dataset.output_size, args.hidden_size, args.n_heads, type1_dataset.oracle_size, type1_dataset.n_examples_per_sample)
    assert model_n_params == estimated_n_params, f"Model has {model_n_params} parameters, but expected {estimated_n_params} parameters."

    print("Number of parameters:", model_n_params)
    print("Hidden size:", args.hidden_size)

    # save the arguments to the directory using torch.save
    torch.save(args, f"{save_dir}/args.pth")


    # train model
    if args.load_dir is None:
        # callbacks to view errors during training
        cb1 = CustomCallback(type1_dataset.data_type, testing_dataset=None, prefix="type1", logdir=save_dir)
        cb2 = CustomCallback(type1_dataset.data_type, testing_dataset=type2_dataset, prefix="type2", tensorboard=cb1.tensorboard)
        cb3 = CustomCallback(type1_dataset.data_type, testing_dataset=type3_dataset, prefix="type3", tensorboard=cb1.tensorboard)
        cb4 = CustomCallback(type1_dataset.data_type, testing_dataset=type4_dataset, prefix="type4", tensorboard=cb1.tensorboard)
        cb_list = ListCallback([cb1, cb2, cb3, cb4])

        # train the model
        model.train_model(type1_dataset, epochs=args.epochs, callback=cb_list)

        # save the model
        torch.save(model.state_dict(), f"{save_dir}/model.pth")

    else: # load the model
        loaded_params = torch.load(args.load_dir, weight_only=True, map_location=args.device)
        model.load_state_dict(loaded_params, strict=True)

    # plot results
    context = torch.no_grad() if args.algorithm != "MAML" else nullcontext()
    with context:
        plotter = get_plotting_function(args.dataset)
        for i, dataset in enumerate([type1_dataset, type2_dataset, type3_dataset, type4_dataset]):
            # get data
            example_xs, example_ys, xs, ys, info = dataset.sample()

            # get predictions
            if type(model) == Oracle:
                y_hats = model.predict_from_examples(example_xs, example_ys, xs, info=info, method="least_squares")
            else:
                y_hats = model.predict_from_examples(example_xs, example_ys, xs, method="least_squares")


            # plot results
            plotter(xs.detach().cpu(), ys.detach().cpu(), y_hats.detach().cpu(), example_xs.detach().cpu(), example_ys.detach().cpu(), save_dir, i, info)


