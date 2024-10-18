from FunctionEncoder import *

from src.datasets.CIFAR100 import get_cifar_datasets, plot_cifar
from src.datasets.Donut import get_donut_datasets
from src.datasets.Polynomial import get_polynomial_datasets, plot_polynomial
from src.datasets.ToyCategorical import get_toy_categetorical_datasets, plot_toy_categorical
from src.datasets.SevenScenes import get_7scenes_datasets, plot_7scenes


def get_datasets(dataset:str, device, n_examples):
    if dataset.lower() == "Polynomial".lower():
        return get_polynomial_datasets(device, n_examples)
    elif dataset.lower() == "Donut".lower():
        return get_donut_datasets(device, n_examples)
    elif dataset.lower() == "CIFAR100".lower() or dataset.lower() == "CIFAR".lower():
        return get_cifar_datasets(device, n_examples)
    elif dataset.lower() == "Categorical".lower():
        return get_toy_categetorical_datasets(device, n_examples)
    elif dataset.lower() == "7Scenes".lower():
        return get_7scenes_datasets(device, n_examples)
    else:
        raise ValueError(f"Dataset {dataset} not recognized.")


def get_plotting_function(dataset:str):
    if dataset.lower() == "Polynomial".lower():
        return plot_polynomial
    elif dataset.lower() == "CIFAR100".lower() or dataset.lower() == "CIFAR".lower():
        return plot_cifar
    elif dataset.lower() == "Categorical".lower():
        return plot_toy_categorical
    elif dataset.lower() == "7Scenes".lower():
        return plot_7scenes
    else:
        raise ValueError(f"Dataset {dataset} not recognized.")