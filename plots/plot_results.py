import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from names import *

if __name__ == "__main__":
    algs = "LS IP AE Transformer TFE Oracle BFB BF MAML1 MAML5 Siamese Proto".split(" ")
    datasets = "Polynomial CIFAR 7Scenes Ant".split(" ")
    logdir = "logs"

    for dataset in datasets:
        if not os.path.exists(os.path.join(logdir, dataset)):
            print(f"Skipping {dataset} because it does not exist.")
            print("If you think this is an error, run collect_tensorboards.py first.")
            continue
        print(f"Plotting {dataset}")
        data = torch.load(f"{logdir}/{dataset}/data.pth", weights_only=False)

        # for categorical data, accuracy is the ultimate metric
        if dataset == "CIFAR":
            tags = [ "train/accuracy", "type1/accuracy", "type3/accuracy"]
        elif dataset == "7Scenes":
            tags = [ "train/mean_distance_squared", "type1/mean_distance_squared", "type3/mean_distance_squared"]
        else:
            tags = [ "train/mean_distance_squared", "type1/mean_distance_squared", "type2/mean_distance_squared", "type3/mean_distance_squared"]

        # create plots
        if len(tags) == 4:
            fig, ax = plt.subplots(1, 4, figsize=(24, 6))
        elif len(tags) == 3:
            fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        axs = ax.flatten()


        # plot each tag
        for i, tag in enumerate(tags):
            title = titles[tag]
            yax = yaxis[tag]
            axs[i].set_title(title)
            axs[i].set_xlabel("Epoch")
            if i == 0:
                axs[i].set_ylabel(yax)

            if "accuracy" in tag:
                axs[i].set_ylim([0.4, 1.0])
                axs[i].set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                axs[i].set_yticklabels(["40%", "50%", "60%", "70%", "80%", "90%", "100%"])
            else:
                axs[i].set_yscale("log")

            # plot each alg for each tag
            for alg in algs:
                if alg not in data:
                    continue
                if alg == "MAML5":
                    continue # TEMP TODO remove

                # gather the median, min, and max of all seeds
                seeds = list(data[alg].keys())
                if len(seeds) == 0:
                    continue
                values = np.array([data[alg][seed][tag] for seed in seeds])
                median = np.median(values, axis=0)
                min_val = np.min(values, axis=0)
                max_val = np.max(values, axis=0)
                axs[i].plot(median, label=alg_names[alg], color=alg_colors[alg])
                axs[i].fill_between(range(len(median)), min_val, max_val, color=alg_colors[alg], alpha=0.2)
        axs[i].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(logdir, dataset, "results.png"))




