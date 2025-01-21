import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from names import *

# set font size and type
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams.update({'font.family': FONT})

def smooth(ys, size=20):
    """smooths a list of values"""
    return np.array([np.mean(ys[max(0, i-size):i+size]) for i in range(len(ys))])



def safe_med_min_min(values):
    """ Returns the median, min, and max of each column with the nans removed"""
    medians, mins, maxs = [], [], []
    for col in range(values.shape[1]):
        data = values[:, col]
        data = data[~np.isnan(data)]
        data = data[~np.isinf(data)]
        if len(data) <= 3:
            continue
        median = np.median(data)
        medians.append(median)


        minn = np.min(data)
        mins.append(minn)


        maxx = np.max(data)
        maxs.append(maxx)


    medians, mins, maxs = np.array(medians), np.array(mins), np.array(maxs)
    return medians, mins, maxs


if __name__ == "__main__":
    alg = "LS"
    datasets = "Polynomial CIFAR 7Scenes Ant".split(" ")
    logdir = "logs/basis"
    n_basis = "1 2 3 5 10 20 40 60 80 100".split(" ")
    n_basis = [int(n) for n in n_basis]

    for dataset in datasets:
        if not os.path.exists(os.path.join(logdir, dataset)):
            print(f"Skipping {dataset} because it does not exist.")
            print("If you think this is an error, run collect_tensorboards_basis.py first.")
            continue
        print(f"Plotting {dataset}")
        try:
            data = torch.load(f"{logdir}/{dataset}/data.pth", weights_only=False)
        except FileNotFoundError:
            print(f"Skipping {dataset} because it does not exist.")
            print("If you think this is an error, run collect_tensorboards_basis.py first.")
            continue
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
            fig, ax = plt.subplots(1, 3, figsize=(24, 6))
        axs = ax.flatten()


        # plot each tag
        for i, tag in enumerate(tags):
            title = titles[tag]
            yax = yaxis[tag] + "\n(After 50k Gradient Steps)"
            axs[i].set_title(title)
            axs[i].set_xlabel("Number of Basis Functions")
            if i == 0:
                axs[i].set_ylabel(yax)

            if "accuracy" in tag:
                axs[i].set_ylim([0.4, 1.0])
                axs[i].set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                axs[i].set_yticklabels(["40%", "50%", "60%", "70%", "80%", "90%", "100%"])
            else:
                maybe_lims = ylims[dataset][title]
                if maybe_lims is not None: # occasionally maml5 breaks and creates outliers. We shorten the plots to ignore these. 
                    axs[i].set_ylim(maybe_lims)
                axs[i].set_yscale("log")

            # plot each alg for each tag
            mins, maxs, meds = [], [], []
            for n_bas in n_basis:
                if n_bas not in data:
                    continue
                # gather the median, min, and max of all seeds
                seeds = list(data[n_bas].keys())
                if len(seeds) == 0:
                    continue
                values = [data[n_bas][seed][tag] for seed in seeds]
                shortest = min([len(v) for v in values])
                assert shortest >= 50000, f"One alg did not finish training. len={shortest}, should be 50000"
                values = np.array([v[-50:].mean() for v in values])
                minn, maxx, median = np.min(values), np.max(values), np.median(values)
                mins.append(minn)
                maxs.append(maxx)
                meds.append(median)
                




            axs[i].plot(n_basis, meds, color=alg_colors[alg])
            axs[i].fill_between(n_basis, mins, maxs, color=alg_colors[alg], alpha=0.2, lw=0)
               
        
        plt.tight_layout()
        plt.savefig(os.path.join(logdir, dataset, f"basis_{dataset}.png"))




