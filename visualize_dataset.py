import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torchvision.utils import make_grid


parser = ArgumentParser("Visualize images from the dataset")

parser.add_argument("-d", "--data-path", type=str, required=True,
                    help="path to the directory where dataset files are stored")


def create_image_grid(sample_images):
    grid_images = make_grid(sample_images, nrow=3)
    grid_images = grid_images.numpy().transpose((1, 2, 0))
    return grid_images


def plot_grid(images_grid):
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(images_grid)
    plt.title("Sample images from dataset")
    plt.axis("off")
    plt.show()


def main():
    args = parser.parse_args()

    train_set_path = os.path.join(args.data_path, "train_images.npy")
    train_set = np.load(train_set_path)
    indices = torch.randint(train_set.shape[0], (9,))
    sample_subset = torch.from_numpy(train_set[indices])
    images_grid = create_image_grid(sample_subset)
    plot_grid(images_grid)


if __name__ == "__main__":
    main()
