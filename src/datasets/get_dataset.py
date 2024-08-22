from FunctionEncoder import *

from src.datasets.Donut import get_donut_datasets
from src.datasets.Polynomial import get_polynomial_datasets, plot_polynomial


def get_datasets(dataset:str, device, n_examples):
    if dataset == "Polynomial":
        return get_polynomial_datasets(device, n_examples)
    elif dataset == "Donut":
        return get_donut_datasets(device, n_examples)
    else:
        raise ValueError(f"Dataset {dataset} not recognized.")


def get_plotting_function(dataset:str):
    if dataset == "Polynomial":
        return plot_polynomial