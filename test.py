import argparse
import os
from contextlib import nullcontext
from typing import Union

from datetime import datetime
import torch
from FunctionEncoder import *
import time

from src.CustomCallback import CustomCallback
from src.algs.PrototypicalNetwork import ProtoTypicalNetwork
from src.algs.get_model import get_model, predict_number_params, get_number_params, get_hidden_size
from src.datasets.get_dataset import get_datasets, get_plotting_function
from src.algs.Oracle import Oracle
from src.algs.Transformer import get_gradient_accumulation_steps


def load_args(args):
    load_dir = args.load_dir
    args = torch.load(f"{load_dir}/args.pth")
    args.load_dir = load_dir
    return args

def check_args(args):
    acceptable_algs = ["FE", "AE", "Transformer", "TFE", "Oracle", "BF", "BFB", "MAML1", "MAML5", "LS", "IP", "TransformerWithPositions", "Siamese", "Proto", "MLP"]
    acceptable_datasets = ["Polynomial", "Donut", "CIFAR", "Categorical", "7Scenes", "Ant"]
    assert args.n_basis >=1, f"n_basis must be at least 1, got {args.n_basis}"
    assert args.n_examples >= 1, f"n_examples must be at least 1, got {args.n_examples}"
    assert args.epochs >= 1, f"epochs must be at least 1, got {args.epochs}"
    assert args.algorithm in acceptable_algs, f"algorithm must be in f{acceptable_algs}, got {args.algorithm}"
    assert args.dataset in acceptable_datasets, f"dataset must be in {acceptable_datasets}, got {args.dataset}"
    assert args.n_params >= 1, f"n_params must be at least 1, got {args.n_params}"
    assert args.n_layers >= 1, f"n_layers must be at least 1, got {args.n_layers}"
    assert args.device in ["cpu", "cuda"] + [f"cuda:{i}" for i in range(8)], f"device must be in ['cpu', 'cuda'] or an integer, got {args.device}"
    if args.algorithm == "Siamese":
        if not args.dataset == "CIFAR":
            print("Siamese algorithm only works with classificiation datasets (CIFAR)")
            exit()
    if args.algorithm == "Proto":
        if not args.dataset == "CIFAR":
            print("Prototypical networks algorithm only works with classificiation datasets (CIFAR)")
            exit()
    if args.cross_entropy:
        if not args.dataset == "CIFAR":
            print("Cross entropy loss only works with classification datasets (CIFAR)")
            exit()
        if args.algorithm in ["LS", "IP"]:
            print("Function Encoders require inner product based loss functions, not cross entropy.")
            exit()
        if args.algorithm in ["Siamese", "Proto"]:
            print("Siamese and Proto algorithms have their own custom loss functions, cannot use cross entropy.")
            exit()

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
    parser.add_argument("--cross_entropy", action="store_true")
    parser.add_argument("--maml_internal_learning_rate", type=float, default=None) # defaults to pretuned parameters for each dataset
    args = parser.parse_args()

    # find device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device not in ["cpu", "cuda"]:
        args.device = f"cuda:{args.device}"

    # load arguments if they exist
    if args.load_dir is not None:
        args = load_args(args)
    if args.device == "cpu":
        print("WARNING: Running on CPU. This will be slow.")

    # check arguments for valid settings only.
    check_args(args)

    # If MAML, the number after MAML is the number of inner steps
    if "MAML" in args.algorithm:
        maml_steps = int(args.algorithm.split("MAML")[1])
    else:
        maml_steps = 0

    # set gradient accumulation for transformer since it devours memory
    if args.algorithm == "Transformer":
        args.gradient_accumulation = get_gradient_accumulation_steps(args.dataset)
        args.num_functions = 10 // args.gradient_accumulation
        args.epochs = args.epochs * args.gradient_accumulation
    else:
        args.gradient_accumulation = 1
        args.num_functions = 10 # total number of functions per grad step is the same for both!
    assert args.gradient_accumulation * args.num_functions == 10, "Total number of functions per gradient step must be 10."

    # set seed
    torch.manual_seed(args.seed)

    # create save directory
    if args.load_dir is None:
        datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        alg_save_name = args.algorithm
        if args.algorithm == "MAML":
            alg_save_name += f'_{args.maml_steps}'
        save_dir = f"{args.log_dir}/{args.dataset}/{alg_save_name}/{datetime}"
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = args.load_dir


    # get datasets
    train_dataset, type1_dataset, type2_dataset, type3_dataset = get_datasets(args.dataset, args.device, args.n_examples, n_functions=args.num_functions)

    # get model
    args.hidden_size = get_hidden_size(args.algorithm, type1_dataset, args.n_params, args.n_basis, args.n_layers, args.n_heads) # calculated to have approximately args.n_params
    model = get_model(args.algorithm, type1_dataset, args.n_basis, args.n_layers, args.n_heads, args.hidden_size, maml_steps=maml_steps, cross_entropy=args.cross_entropy, maml_internal_learning_rate=args.maml_internal_learning_rate, device=args.device, gradient_accumulation=args.gradient_accumulation)

    # assert right size
    model_n_params = get_number_params(model)
    model_type = model.model_type
    model_kwargs = {"hidden_size": args.hidden_size, "n_layers": args.n_layers, "n_heads": args.n_heads, "oracle_size": type1_dataset.oracle_size, "n_examples": type1_dataset.n_examples}
    estimated_n_params = predict_number_params(args.algorithm, args.n_basis, type1_dataset.input_size, type1_dataset.output_size, model_type, model_kwargs)
    # assert model_n_params == estimated_n_params, f"Model has {model_n_params} parameters, but expected {estimated_n_params} parameters."
    print("Running ", args.algorithm, " on ", args.dataset)
    print("Number of parameters:", model_n_params)
    print("Hidden size:", args.hidden_size)

    # save the arguments to the directory using torch.save
    torch.save(args, f"{save_dir}/args.pth")


    # train model
    if args.load_dir is None:
        # callbacks to view errors during training
        cb_list = []
        cb_list.append(CustomCallback(train_dataset.data_type, testing_dataset=None, prefix="train", logdir=save_dir))
        cb_list.append(CustomCallback(train_dataset.data_type, testing_dataset=type1_dataset, prefix="type1", tensorboard=cb_list[0].tensorboard))
        if type2_dataset: # some datasets don't have type2
            cb_list.append(CustomCallback(train_dataset.data_type, testing_dataset=type2_dataset, prefix="type2", tensorboard=cb_list[0].tensorboard))
        cb_list.append(CustomCallback(train_dataset.data_type, testing_dataset=type3_dataset, prefix="type3", tensorboard=cb_list[0].tensorboard))
        if args.algorithm in ["LS", "IP", "FE"]: # note you can do this for debugging purposes.
            cb_list.append(TensorboardCallback(tensorboard=cb_list[0].tensorboard, prefix="fe_debug"))
        cb_list = ListCallback(cb_list)

        # train the model
        model.train_model(train_dataset, epochs=args.epochs, callback=cb_list)

        # save the model
        torch.save(model.state_dict(), f"{save_dir}/model.pth")

    else: # load the model
        loaded_params = torch.load(args.load_dir, weight_only=True, map_location=args.device)
        model.load_state_dict(loaded_params, strict=True)

    # plot results
    context = torch.no_grad() if not "MAML" in args.algorithm else nullcontext()
    with context:
        plotter = get_plotting_function(args.dataset)
        for i, dataset in enumerate([type1_dataset, type2_dataset, type3_dataset]):
            if dataset is None:
                continue

            # get data
            dataset.n_examples = 10000
            example_xs, example_ys, xs, ys, info = dataset.sample()

            # if using function encoder, estimate L2 distance
            if args.algorithm in ["FE", "IP", "LS"]:
                l2_distance = model.estimate_L2_error(example_xs, example_ys)

                l2_distance_2 = model._distance(example_ys, model.predict_from_examples(example_xs, example_ys, example_xs, method="least_squares"), squared=False)

                coeffs, gram = model.compute_representation(example_xs, example_ys, method="least_squares")
                f_hat_norm_squared = torch.einsum("fk,fkl,fl->f", coeffs, gram, coeffs)
                l2_distance_3 = model._norm(example_ys, squared=True) - 2 * model._inner_product(example_ys, model.predict_from_examples(example_xs, example_ys, example_xs, method="least_squares")) + f_hat_norm_squared
                l2_distance_3 = torch.sqrt(l2_distance_3)
                print(f"L2 distance: {l2_distance}")
                print(f"L2 distance 2: {l2_distance_2}")
                print(f"L2 distance 3: {l2_distance_3}")
                print()
                info["L2_distance"] = l2_distance

            # get predictions
            if type(model) == Oracle or type(model) == ProtoTypicalNetwork:
                y_hats = model.predict_from_examples(example_xs, example_ys, xs, info=info, method="least_squares")
            else:
                y_hats = model.predict_from_examples(example_xs, example_ys, xs, method="least_squares")


            # plot results
            plotter(xs.detach().cpu(), ys.detach().cpu(), y_hats.detach().cpu(), example_xs.detach().cpu(), example_ys.detach().cpu(), save_dir, i, info)
            del example_xs, example_ys, xs, ys, info, y_hats # frees memory for MAML

    print()
