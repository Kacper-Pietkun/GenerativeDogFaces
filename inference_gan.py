import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from utils import get_device, Visualizator
from gan_model import GAN


parser = ArgumentParser("Inference for VAE model - generate new random images using trained decoder")

parser.add_argument("--model-path", type=str, required=True,
                    help="Path to the model state dict")

parser.add_argument("--save-path", type=str, required=True,
                    help="Path to directory where generated images will be saved")

parser.add_argument("--noise-dimension", type=int, default=100,
                    help="Size of the generator's noise input")

parser.add_argument("--gpu", type=int, default=1,
                    help="1 - use gpu, 0 - use cpu")


def main():
    args = parser.parse_args()
    device = get_device(args.gpu)

    visualizator = Visualizator(args.save_path, use_tanh=True)
    gan = GAN(3, args.noise_dimension, device).to(device)
    gan.load_state_dict(torch.load(args.model_path))
    gan.eval()

    trained_generator = gan.generator
    with torch.no_grad():
        out = trained_generator.generate(64)
    visualizator.plot_images_inference(out, show=True, rows=8, cols=8)


if __name__ == "__main__":
    main()
