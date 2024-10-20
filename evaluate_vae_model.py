import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from vae_model import VAE
from torchvision.utils import make_grid


parser = ArgumentParser("Evaluate VAE model - reconstruct images, and generate new random images")

parser.add_argument("--dataset-path", type=str, required=True,
                    help="Path to the directory where train, val, test sets are stored")

parser.add_argument("--model-path", type=str, required=True,
                    help="Path to the model state dict")

parser.add_argument("--embedding-size", type=int, default=128,
                    help="size of the VAE's embedding vector")

parser.add_argument("--gpu", type=int, default=1,
                    help="1 - use gpu, 0 - use cpu")


def create_image_grid(sample_images):
    grid_images = make_grid(sample_images, nrow=3)
    grid_images = grid_images.numpy().transpose((1, 2, 0))
    return grid_images


def get_device(use_gpu):
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:0")
        else:
            raise ValueError("GPU training was chosen but cuda is not available")
    else:
        device = torch.device("cpu")
    return device


def plot_reconstruction(original, reconstruction, set_name):
    original_grid = create_image_grid(original)
    reconstructed_grid = create_image_grid(reconstruction.cpu())

    plt.figure(figsize=(14, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(original_grid)
    plt.title(f"Original - {set_name}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_grid)
    plt.title("Reconstruction")
    plt.axis("off")
    plt.show()



def main():
    args = parser.parse_args()
    device = get_device(args.gpu)

    vae = VAE(3, args.embedding_size, device).to(device)
    vae.load_state_dict(torch.load(args.model_path))
    vae.eval()

    train_set_path = os.path.join(args.dataset_path, "train_images.npy")
    train_set = torch.from_numpy(np.load(train_set_path))

    validation_set_path = os.path.join(args.dataset_path, "validation_images.npy")
    validation_set = torch.from_numpy(np.load(validation_set_path))

    test_set_path = os.path.join(args.dataset_path, "test_images.npy")
    test_set = torch.from_numpy(np.load(test_set_path))


    original_train = train_set[:9]
    reconstructed_train, _, _ = vae(original_train.to(device))
    plot_reconstruction(original_train, reconstructed_train, "train set")


    original_validation = validation_set[:9]
    reconstructed_validation, _, _ = vae(original_validation.to(device))
    plot_reconstruction(original_validation, reconstructed_validation, "validation set")


    original_test = test_set[:9]
    reconstructed_test, _, _ = vae(original_test.to(device))
    plot_reconstruction(original_test, reconstructed_test, "test set")
    

    # get only the decoder for generating new images
    trained_decoder = vae.decoder
    sample_means = torch.zeros(9 * args.embedding_size)
    sample_stds = torch.ones(9 * args.embedding_size)
    sample_gauss = torch.normal(sample_means, sample_stds).reshape(9, -1).to(device)
    out = trained_decoder(sample_gauss)
    plot_reconstruction(out.cpu(), out, "new")



if __name__ == "__main__":
    main()


