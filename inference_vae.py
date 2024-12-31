import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils import get_device, Visualizator
from vae_model import VAE


parser = ArgumentParser("Inference for GAN model - generate new random images using trained generator")

parser.add_argument("--model-path", type=str, required=True,
                    help="Path to the model state dict")

parser.add_argument("--save-path", type=str, required=True,
                    help="Path to directory where generated images will be saved")

parser.add_argument("--embedding-size", type=int, default=128,
                    help="size of the VAE's embedding vector")

parser.add_argument("--gpu", type=int, default=1,
                    help="1 - use gpu, 0 - use cpu")


def main():
    args = parser.parse_args()
    device = get_device(args.gpu)

    visualizator = Visualizator(args.save_path, use_tanh=False)
    vae = VAE(3, args.embedding_size, device).to(device)
    vae.load_state_dict(torch.load(args.model_path))
    vae.eval()

    trained_decoder = vae.decoder
    sample_means = torch.zeros(64 * args.embedding_size)
    sample_stds = torch.ones(64 * args.embedding_size)
    sample_gauss = torch.normal(sample_means, sample_stds).reshape(64, -1).to(device)
    with torch.no_grad():
        out = trained_decoder(sample_gauss)
    visualizator.plot_images_inference(out, show=True, rows=8, cols=8)


if __name__ == "__main__":
    main()
