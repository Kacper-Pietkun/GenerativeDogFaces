import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torchvision.utils import make_grid
from torchvision.transforms import v2


parser = ArgumentParser("Visualize images from the dataset")

parser.add_argument("--data-path", type=str, required=True,
                    help="path to the directory where dataset files are stored")


def create_image_grid(sample_images):
    grid_images = make_grid(sample_images, nrow=3)
    grid_images = grid_images.numpy().transpose((1, 2, 0))
    return grid_images


def get_augmentations(args):
    transforms = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.RandomAdjustSharpness(sharpness_factor=3, p=0.5),
        v2.RandomEqualize(p=0.5),
        v2.RandomAutocontrast(0.5),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True)
    ])
    return transforms


def plot_grid(images_grid, augmented_images_grid):
    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(images_grid)
    plt.title("Sample images from dataset")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(augmented_images_grid)
    plt.title("augmented images from dataset")
    plt.axis("off")
    plt.show()
    plt.close()


def apply_augmentations(args, original_images):
    augmentations = get_augmentations(args)
    augmented_images = []
    for image in original_images:
        augmented_images.append(augmentations(image))
    return torch.stack(augmented_images)


def main():
    args = parser.parse_args()

    train_set_path = os.path.join(args.data_path, "train_images.npy")
    train_set = np.load(train_set_path)
    for i in range(5):
        indices = torch.randint(train_set.shape[0], (9,))
        sample_subset = torch.from_numpy(train_set[indices])
        augmented_sample_subset = apply_augmentations(args, sample_subset)
        images_grid = create_image_grid(sample_subset)
        augmented_images_grid = create_image_grid(augmented_sample_subset)
        plot_grid(images_grid, augmented_images_grid)


if __name__ == "__main__":
    main()
