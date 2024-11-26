import os
import sys

import matplotlib
import numpy as np
import torch
from FunctionEncoder import BaseDataset
from PIL import Image
from PIL.Image import Resampling
from matplotlib import pyplot as plt


def download_datasets():
    # img size after reshaping
    img_size = (40, 30)


    # fetch location of current file
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)

    # create a subdir called 7ScenesData
    data_dir = os.path.join(current_dir, "7ScenesData")
    # if data dir exists, skip
    if os.path.exists(data_dir):
        print("7Scenes data already downloaded")
        return
    os.makedirs(data_dir, exist_ok=True)

    # download the data
    links = ["http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip",
             "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/fire.zip",
             "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip",
             "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip",
             "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/pumpkin.zip",
             "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip",
             "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/stairs.zip"]
    for link in links:
        # downloads
        os.system(f"wget {link} -P {data_dir} -q --show-progress")

        # unzip and remove zip
        os.system(f"unzip {os.path.join(data_dir, link.split('/')[-1])} -d {data_dir} > /dev/null")
        os.system(f"rm {os.path.join(data_dir, link.split('/')[-1])} > /dev/null")

        # for each file in the new dir, if its a zip, unzip it also
        subdir_name = link.split('/')[-1].split(".")[0]
        subdir = os.path.join(data_dir, subdir_name)
        zips = os.listdir(subdir)
        zips.sort()
        for file in zips:
            if file.endswith(".zip"):
                # unzip and delete the zip
                print("Unzipping",  file, end=".\t")
                os.system(f"unzip {os.path.join(subdir, file)} -d {subdir} > /dev/null")
                os.system(f"rm {os.path.join(subdir, file)} > /dev/null")

                # now this subdir contains a bunch of images. First of all, delete all of the *depth*
                subsubdir = os.path.join(subdir, file.replace(".zip", ""))
                os.system(f"rm {os.path.join(subdir, subsubdir, '*depth*')}  > /dev/null")

                # next load all images, resize them, then save them as a tensor
                # at the same time, process the labels for consistency
                images = os.listdir(subsubdir)
                images = [image for image in images if image.endswith(".color.png")]
                images.sort()
                labels = [image.replace(".color.png", ".pose.txt") for image in images]

                # for each image, load, resize, save
                image_list = []
                label_list = []
                print(f"Converting {file.replace('.zip', '')} to tensor...")
                for i in range(len(images)):
                    # load image and label
                    image_path = os.path.join(subdir, subsubdir, images[i])
                    label_path = os.path.join(subdir, subsubdir, labels[i])

                    # resize image
                    image = Image.open(image_path)
                    image = image.resize(img_size, resample=Resampling.BOX)

                    # convert to tensor
                    image = torch.tensor(image.getdata()).reshape(img_size[1], img_size[0], 3)
                    image = image.permute(2, 0, 1).float() / 255.0

                    # load label, convert to tensor
                    label = np.loadtxt(label_path)
                    label = torch.tensor(label).float()

                    # store
                    image_list.append(image)
                    label_list.append(label)

                # convert to one big tensor
                image_tensor = torch.stack(image_list)
                label_tensor = torch.stack(label_list)

                # save
                torch.save(image_tensor, os.path.join(subdir, subsubdir, "images.pt"))
                torch.save(label_tensor, os.path.join(subdir, subsubdir, "labels.pt"))

                # delete the images and labels
                os.system(f"rm {os.path.join(subdir, subsubdir, '*.color.png')} > /dev/null")
                os.system(f"rm {os.path.join(subdir, subsubdir, '*.pose.txt')} > /dev/null")
    print("Downloaded 7Scenes data to src/datasets/7ScenesData")


if __name__ == "__main__":
    download_datasets()


class SevenScenesDataset(BaseDataset):
    def __init__(self,
                 split="train", # train or test. Test is type 1 transfer
                 heldout=False,  # heldout scenes are type 3 transfer
                 heldout_scene="redkitchen",  # the heldout scene
                 **kwargs):
        assert split in ["train", "test"], "split must be 'train' or 'test'"

        super().__init__(input_size=(3,30,40),
                         output_size=(3,), # (16,),
                         n_queries=200,
                         data_type="deterministic",
                         **kwargs)
        self.split = split
        self.heldout = heldout
        self.heldout_scene = heldout_scene
        self.oracle_size = self.n_functions

        # load the data from the 7ScenesData directory
        scenes_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "7ScenesData")

        # get all the scenes
        if not heldout:
            self.scenes = os.listdir(scenes_dir)
            self.scenes = [scene for scene in self.scenes if scene != self.heldout_scene]
        else:
            self.scenes = [self.heldout_scene]

        # get all the sequences for each scene
        self.scenes_sequencies = {}
        for scene in self.scenes:
            self.scenes_sequencies[scene] = os.listdir(os.path.join(scenes_dir, scene))
            self.scenes_sequencies[scene] = [sequence for sequence in self.scenes_sequencies[scene] if os.path.isdir(os.path.join(scenes_dir, scene, sequence))]
            self.scenes_sequencies[scene].sort()
            # if we are using the heldout scene, might as well test on all sequences.
            # otherwise, train on all but the last sequence.
            if not self.heldout:
                if self.split == "train":
                    self.scenes_sequencies[scene] = self.scenes_sequencies[scene][:-1]
                else:
                    self.scenes_sequencies[scene] = self.scenes_sequencies[scene][-1:]

        # get all the images for each sequence
        self.scenes_sequences_images = {}
        for scene in self.scenes:
            self.scenes_sequences_images[scene] = {}
            for sequence in self.scenes_sequencies[scene]:
                self.scenes_sequences_images[scene][sequence] = {}
                self.scenes_sequences_images[scene][sequence]["images"] = torch.load(os.path.join(scenes_dir, scene, sequence, "images.pt"), weights_only=True, map_location="cpu")
                self.scenes_sequences_images[scene][sequence]["poses"] = torch.load(os.path.join(scenes_dir, scene, sequence, "labels.pt"), weights_only=True, map_location="cpu")

        # oracle information, hardcoded so its consistent
        self.oracle_index = {"chess":
                                 {"seq-01": 0,
                                  "seq-02": 1,
                                  "seq-03": 2,
                                  "seq-04": 3,
                                  "seq-05": 4,
                                  "seq-06": 5,
                                  },
                                "fire":
                                    {"seq-01": 6,
                                    "seq-02": 7,
                                    "seq-03": 8,
                                    "seq-04": 9,
                                    },
                                "heads":
                                    {"seq-01": 10,
                                    "seq-02": 11,
                                     },
                                "office":
                                    {"seq-01": 12,
                                    "seq-02": 13,
                                    "seq-03": 14,
                                    "seq-04": 15,
                                    "seq-05": 16,
                                    "seq-06": 17,
                                    "seq-07": 18,
                                    "seq-08": 19,
                                    "seq-09": 20,
                                    "seq-10": 21,
                                     },
                                "pumpkin":
                                    {"seq-01": 22,
                                    "seq-02": 23,
                                    "seq-03": 24,
                                    "seq-04": 25,
                                    "seq-05": 26,
                                    "seq-06": 27,
                                    "seq-07": 28,
                                    "seq-08": 29,
                                     },
                                "redkitchen":
                                    {"seq-01": 30,
                                    "seq-02": 31,
                                    "seq-03": 32,
                                    "seq-04": 33,
                                    "seq-05": 34,
                                    "seq-06": 35,
                                    "seq-07": 36,
                                    "seq-08": 37,
                                    "seq-09": 38,
                                    "seq-10": 39,
                                    "seq-11": 40,
                                    "seq-12": 41,
                                    "seq-13": 42,
                                    "seq-14": 43,
                                     },
                                "stairs":
                                    {"seq-01": 44,
                                    "seq-02": 45,
                                    "seq-03": 46,
                                    "seq-04": 47,
                                    "seq-05": 48,
                                    "seq-06": 49,
                                     },
                                }
        self.oracle_size = 50


    def sample_info(self):
        scene_indicies = torch.randint(0, len(self.scenes), (self.n_functions,))
        num_sequences = [len(self.scenes_sequencies[self.scenes[i]]) for i in scene_indicies]
        sequence_indicies = torch.cat([torch.randint(0, num_sequences[i], (1,)) for i in range(self.n_functions)], dim=0)
        scenes = [self.scenes[i] for i in scene_indicies]
        sequences = [self.scenes_sequencies[scenes[i]][sequence_indicies[i]] for i in range(self.n_functions)]
        oracle_indicies = [self.oracle_index[scenes[i]][sequences[i]] for i in range(self.n_functions)]
        oracle_ohe = torch.nn.functional.one_hot(torch.tensor(oracle_indicies), num_classes=self.oracle_size).float()
        return {"scene_indicies": scene_indicies,
                "sequence_indicies": sequence_indicies,
                "scenes": scenes,
                "sequences": sequences,
                "oracle_inputs": oracle_ohe.to(self.device),
                }

    def sample_images(self, info, count):
        all_images = []
        all_labels = []
        for i in range(self.n_functions):
            # for this scene and sequence
            scene = info["scenes"][i]
            sequence = info["sequences"][i]

            # sample random images
            image_indicies = torch.randint(0, len(self.scenes_sequences_images[scene][sequence]["images"]), (count,))

            # get images
            seq_images = self.scenes_sequences_images[scene][sequence]["images"][image_indicies]
            seq_labels = self.scenes_sequences_images[scene][sequence]["poses"][image_indicies]

            # save the images and labels
            all_labels.append(seq_labels)
            all_images.append(seq_images)
        all_images = torch.stack(all_images)
        all_labels = torch.stack(all_labels)
        return all_images, all_labels

    def sample(self):
        with torch.no_grad():
            # select scenes and sequences
            info = self.sample_info()

            # get images and labels
            example_xs, example_ys = self.sample_images(info, self.n_examples)
            xs, ys = self.sample_images(info, self.n_queries)

            # collapse labels from 4x4 to 16
            # example_ys = example_ys.reshape(example_ys.shape[0], example_ys.shape[1], -1)
            # ys = ys.reshape(ys.shape[0], ys.shape[1], -1)

            # get only xyz pos from the matrix
            example_ys = example_ys[..., :3, -1]
            ys = ys[..., :3, -1]

            # change to cuda
            example_xs = example_xs.to(self.device)
            example_ys = example_ys.to(self.device)
            xs = xs.to(self.device)
            ys = ys.to(self.device)

            return example_xs, example_ys, xs, ys, info



def get_7scenes_datasets(device, n_examples, n_functions):
    train = SevenScenesDataset(device=device, n_examples=n_examples, split="train", heldout=False, n_functions=n_functions)
    type1 = SevenScenesDataset(device=device, n_examples=n_examples, split="test", heldout=False, n_functions=n_functions)
    type2 = None  # no linear combinations of distributions
    type3 = SevenScenesDataset(device=device, n_examples=n_examples, heldout=True, n_functions=n_functions)
    return train, type1, type2, type3

def plot_7scenes(xs, ys, y_hats, example_xs, example_ys, save_dir, type_i, info):
    fig, ax = plt.subplots(4, 9, figsize=(36, 16), gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.2, 1, 1, 1, 1, ]})
    for row in range(min(4, xs.shape[0])):
        # example data
        for col in range(4):
            ax[row, col].axis("off")
            img = example_xs[row, col].permute(1, 2, 0).cpu().numpy()
            ax[row, col].imshow(img)
            pos = example_ys[row, col]
            class_name = f"x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}"
            ax[row, col].set_title(class_name)

        # evaluation data
        for col in range(5, 9):
            ax[row, col].axis("off")
            img = xs[row, col-5].permute(1, 2, 0).cpu().numpy()
            ax[row, col].imshow(img)
            pos = ys[row, col-5]
            estimated_pos = y_hats[row, col-5]
            class_name = (f"GroundTruth: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}\n"
                          f"Estimated:   x={estimated_pos[0]:.2f}, y={estimated_pos[1]:.2f}, z={estimated_pos[2]:.2f}")
            ax[row, col].set_title(class_name)

        # disable axis for the two unfilled plots
        ax[row, 4].axis("off")

    # add dashed lines between example and evaluation data
    left = ax[0, 3].get_position().xmax
    right = ax[0, 5].get_position().xmin
    xpos = (left + right) / 2
    top = ax[0, 3].get_position().ymax + 0.05
    bottom = ax[3, 3].get_position().ymin
    line1 = matplotlib.lines.Line2D((xpos, xpos), (bottom, top), transform=fig.transFigure, color="black", linestyle="--")

    fig.lines = line1,

    # add one text above positive samples
    left = ax[0, 0].get_position().xmin
    right = ax[0, 3].get_position().xmax
    xpos = (left + right) / 2
    ypos = ax[0, 0].get_position().ymax + 0.08
    fig.text(xpos, ypos, "Examples", ha="center", va="center", fontsize=32, weight="bold")

    # add one text above negative samples
    left = ax[0, 5].get_position().xmin
    right = ax[0, 8].get_position().xmax
    xpos = (left + right) / 2
    fig.text(xpos, ypos, "Evaluations", ha="center", va="center", fontsize=32, weight="bold")

    plt.savefig(f"{save_dir}/type{type_i + 1}.png")
    plt.clf()

