import numpy as np
import torch
from tensorboard.backend.event_processing import event_accumulator
import os
import sys






def read_tensorboard(logdir, scalars):
    """returns a dictionary of numpy arrays for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    for s in scalars:
        assert s in ea.Tags()["scalars"], f"{s} not found in event accumulator"
    data = {k: np.array([[e.step, e.value] for e in ea.Scalars(k)]) for k in scalars}
    return data


### Collects tensorboard data and saves to disk for easier loading. ###
### This is because tensorboard is slow. ###
if __name__ == "__main__":

    dataset = "Polynomial"
    algs = ["LS", "LS-Parallel"]
    logdirs = ["logs/experiment","logs/parallel"]
    tags = ["type1/accuracy", "type1/mean_distance_squared",
            "type2/accuracy", "type2/mean_distance_squared",
            "type3/accuracy", "type3/mean_distance_squared",
            "train/accuracy", "train/mean_distance_squared",]




    # for all datasets
    if not os.path.exists(os.path.join(logdirs[0], dataset)):
        print(f" {os.path.join(logdirs[0], dataset)} does not exist.")
        exit()
    if not os.path.exists(os.path.join(logdirs[1], dataset)):
        print(f" {os.path.join(logdirs[1], dataset)} does not exist.")
        exit()
    data = {}
    for alg in algs:
        data[alg] = {}


    for logdir in logdirs:
        for alg in algs:
            if alg == "LS" and logdir == "logs/parallel":
                continue
            if alg == "LS-Parallel" and logdir == "logs/experiment":
                continue

            # get alg dir, maybe skip
            alg_dir = os.path.join(logdir, dataset, alg)
            if not os.path.exists(alg_dir):
                print(f"Skipping {alg_dir} in {dataset} because it does not exist.")
                continue

            # get all subdirs
            subdirs = [d for d in os.listdir(alg_dir) if os.path.isdir(os.path.join(alg_dir, d))]
            for subdir in subdirs:
                print(f"Processing {logdir}/{dataset}/{alg}/{subdir}")
                seed_dir = os.path.join(alg_dir, subdir)
                seed = torch.load(os.path.join(seed_dir, "args.pth"), weights_only=False).seed
                data[alg][seed] = {}

                # get all scalars
                for tag in tags:
                    try:
                        tag_data = read_tensorboard(seed_dir, [tag])
                        data[alg][seed][tag] = np.array(tag_data[tag][:, 1])
                    except:
                        continue

        torch.save(data, os.path.join(logdir, dataset,  f"data.pth"))
        print(f"Saving to {logdirs[1]}/{dataset}/data.pth\n")


