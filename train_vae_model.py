import os
import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ModelSaver, BetaCyclicalAnnealing, TensorDatasetWithAugmentations
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from argparse import ArgumentParser
from vae_model import VAE
from tqdm import tqdm


parser = ArgumentParser("Training script for VAE")

parser.add_argument("--dataset-path", type=str, required=True,
                    help="Path to the directory where train, val, test sets are stored")

parser.add_argument("--save-path", type=str, required=True,
                    help="Path to the directory where results will be saved")

parser.add_argument("--gpu", type=int, default=1,
                    help="1 - use gpu, 0 - use cpu")

parser.add_argument("--epochs", type=int, default=100,
                    help="Number of training epochs")

parser.add_argument("--trial", default=None,
                    help="Used only via optuna_search.py script")

parser.add_argument("--lr", type=float, default=0.001,
                    help="initial learning rate used during training")

parser.add_argument("--embedding-size", type=int, default=128,
                    help="size of the VAE's embedding vector")

parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW", "SGD"],
                    help="optimizer used for model training")

parser.add_argument("--weight-decay", type=float, default=0.0,
                    help="weight decay passed to optimizer")

parser.add_argument("--momentum", type=float, default=0.99,
                    help="momentum passed to optimizer")

parser.add_argument("--batch-size", type=int, default=64,
                    help="training batch size")


def get_device(use_gpu):
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:0")
        else:
            raise ValueError("GPU training was chosen but cuda is not available")
    else:
        device = torch.device("cpu")
    return device


def loss_fn(original_image, predicted_image, mean, log_var, beta):
    """
    A loss functuon consisting of a recostruction factor and
    KL divergence (along with a beta parameter given in advance)
    """
    cur_batch_size = original_image.shape[0]
    reconstruction_loss = nn.MSELoss(reduction="none")(original_image, predicted_image).view(cur_batch_size, -1).sum(dim=-1)
    kl_loss = 1 + log_var - mean**2 - torch.exp(log_var)
    kl_loss = -0.5 * torch.sum(kl_loss, dim=-1)
    return (reconstruction_loss + beta * kl_loss).mean(), reconstruction_loss.mean(), (kl_loss).mean()


def save_visualization_outputs(predicted_images, save_path, epoch, rows=3, cols=4):
    """
    Use to save image results after each training epoch and validation epoch
    """
    plt.ioff()
    fig = plt.figure(figsize=(18, 12))
    for idx, img in enumerate(predicted_images, 1):
        plt.subplot(rows, cols, idx)
        plt.imshow(img.detach().numpy().transpose((1, 2, 0)), cmap='gray')
        plt.axis("off")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f"{save_path}/{epoch}.jpg")
    plt.clf()
    plt.close(fig)


def create_dataloaders(args, augmentations):
    train_set_path = os.path.join(args.dataset_path, "train_images.npy")
    train_set = np.load(train_set_path)
    train_dataset = TensorDatasetWithAugmentations(train_set, augmentations)

    validation_set_path = os.path.join(args.dataset_path, "validation_images.npy")
    validation_set = np.load(validation_set_path)

    test_set_path = os.path.join(args.dataset_path, "test_images.npy")
    test_set = np.load(test_set_path)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def create_optimizer(model, args):
    optimizer = None
    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay,
                               betas=(args.momentum, 0.999))
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay,
                                betas=(args.momentum, 0.999))
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay,
                              momentum=args.momentum)
    else:
        raise RuntimeError(f"Specified optimizer: '{args.optimizer}' is not supported")
    return optimizer

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


def train(args, vae, train_dataloader, val_dataloader, optimizer, device, beta_cyclical_annealing, model_saver):
    iteration = 0
    history = []
    best_val_loss = math.inf
    visualization_train_path = os.path.join(args.save_path, "visualization_train")
    visualization_val_path = os.path.join(args.save_path, "visualization_val")
    for epoch in range(args.epochs):
        vae.train()
        train_loss, train_rec_loss, train_kl_loss = 0, 0, 0
        first_batch = None
        for input_image in tqdm(train_dataloader):
            iteration += 1
            input_image = input_image.to(device)

            optimizer.zero_grad()
            predicted_image, mean, log_var = vae(input_image)
            curr_beta = beta_cyclical_annealing.get_updated_beta(iteration)
            loss, rec_loss, kl_loss = loss_fn(input_image, predicted_image, mean, log_var, curr_beta)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * input_image.size(0)
            train_rec_loss += rec_loss.item() * input_image.size(0)
            train_kl_loss += kl_loss.item() * input_image.size(0)
            if first_batch is None:
                first_batch = predicted_image
        save_visualization_outputs(first_batch[:10].cpu(), visualization_train_path, epoch)

        vae.eval()
        val_loss, val_rec_loss, val_kl_loss = 0, 0, 0
        first_batch = None
        with torch.no_grad():
            for input_image in tqdm(val_dataloader):
                input_image = input_image.to(device)

                predicted_image, mean, log_var = vae(input_image)
                loss, rec_loss, kl_loss = loss_fn(input_image, predicted_image, mean, log_var, 1)

                val_loss += loss.item() * input_image.size(0)
                val_rec_loss += rec_loss.item() * input_image.size(0)
                val_kl_loss += kl_loss.item() * input_image.size(0)
                if first_batch is None:
                    first_batch = predicted_image
        save_visualization_outputs(first_batch[:10].cpu(), visualization_val_path, epoch)
        
        train_loss /= len(train_dataloader.sampler)
        train_rec_loss /= len(train_dataloader.sampler)
        train_kl_loss /= len(train_dataloader.sampler)
        val_loss /= len(val_dataloader.sampler)
        val_rec_loss /= len(val_dataloader.sampler)
        val_kl_loss /= len(val_dataloader.sampler)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_rec_loss": train_rec_loss,
            "train_kl_loss": train_kl_loss,
            "val_loss": val_loss,
            "val_rec_loss": val_rec_loss,
            "val_kl_loss": val_kl_loss,
        })
        model_saver.save(vae.state_dict(), val_loss)
        best_val_loss = min(best_val_loss, val_loss)
        if args.trial:
            args.trial.report(val_loss, epoch) 
        print('Epoch: {} Train Loss: {:.4f} ({:.4f}, {:.4f}) Validation Loss: {:.4f} ({:.4f}, {:.4f}) '.format(epoch, train_loss, train_rec_loss, train_kl_loss,
                                                                                                                val_loss, val_rec_loss, val_kl_loss))
    return history, best_val_loss


def plot_losses(args, history):
    train_losses = [epoch_dict.get("train_loss") for epoch_dict in history]
    val_losses = [epoch_dict.get("val_loss") for epoch_dict in history]
    plt.figure(figsize=(14, 8))
    plt.plot(train_losses, color="r", label="Train loss")
    plt.plot(val_losses, color="b", label="Validation loss")
    plt.title("Total loss over epochs")
    plt.legend()
    plt.savefig(f"{args.save_path}/total_loss.jpg")

    train_losses = [epoch_dict.get("train_rec_loss") for epoch_dict in history]
    val_losses = [epoch_dict.get("val_rec_loss") for epoch_dict in history]
    plt.figure(figsize=(14, 8))
    plt.plot(train_losses, color="r", label="Train rec loss")
    plt.plot(val_losses, color="b", label="Validation rec loss")
    plt.title("Rec loss over epochs")
    plt.legend()
    plt.savefig(f"{args.save_path}/reconstruction_loss.jpg")

    train_losses = [epoch_dict.get("train_kl_loss") for epoch_dict in history]
    val_losses = [epoch_dict.get("val_kl_loss") for epoch_dict in history]
    plt.figure(figsize=(14, 8))
    plt.plot(train_losses, color="r", label="Train KL loss")
    plt.plot(val_losses, color="b", label="Validation KL loss")
    plt.title("KL loss over epochs")
    plt.legend()
    plt.savefig(f"{args.save_path}/kl_loss.jpg")


def main(args):
    device = get_device(args.gpu)
    print(f"DEVICE NAME: {device}")

    vae = VAE(3, args.embedding_size, device).to(device)
    augmentations = get_augmentations(args)
    train_dataloader, val_dataloader, _ = create_dataloaders(args, augmentations)
    optimizer = create_optimizer(vae, args)
    model_saver = ModelSaver(args.save_path)
    beta_cyclical_annealing = BetaCyclicalAnnealing(args.epochs, len(train_dataloader))
    
    history, best_val_loss = train(args, vae, train_dataloader, val_dataloader, optimizer, device, beta_cyclical_annealing, model_saver)
    plot_losses(args, history)

    return best_val_loss


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
