import os
import math
import optuna
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from argparse import ArgumentParser
from torchvision.transforms import v2
from utils import ModelSaver, TensorDatasetWithAugmentations
from gan_model import GAN
import matplotlib.pyplot as plt

#TODO: add MIFID(Memorization Informed Frechet Inception Distance) metric

parser = ArgumentParser("Training script for GAN")

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

parser.add_argument("--lr", type=float, default=0.0002,
                    help="initial learning rate used during training")

parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "AdamW", "SGD"],
                    help="optimizer used for model training")

parser.add_argument("--weight-decay", type=float, default=0.0,
                    help="weight decay passed to optimizer")

parser.add_argument("--momentum", type=float, default=0.5,
                    help="momentum passed to optimizer")

parser.add_argument("--noise-dimension", type=int, default=100,
                    help="Number of training epochs")

parser.add_argument("--batch-size", type=int, default=128,
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

def create_dataloaders(args):
    train_augmentations = get_augmentations(args, is_train=True)
    train_set_path = os.path.join(args.dataset_path, "train_images.npy")
    train_set = np.load(train_set_path)
    train_dataset = TensorDatasetWithAugmentations(train_set, train_augmentations)

    test_augmentations = get_augmentations(args, is_train=False)
    validation_set_path = os.path.join(args.dataset_path, "validation_images.npy")
    validation_set = np.load(validation_set_path)
    validation_dataset = TensorDatasetWithAugmentations(validation_set, test_augmentations)

    test_set_path = os.path.join(args.dataset_path, "test_images.npy")
    test_set = np.load(test_set_path)
    test_dataset = TensorDatasetWithAugmentations(test_set, test_augmentations)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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

def get_augmentations(args, is_train):
    if is_train:
        transforms = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomAdjustSharpness(sharpness_factor=3, p=0.5),
            v2.RandomEqualize(p=0.5),
            v2.RandomAutocontrast(0.5),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(lambda x: x * 2 - 1) 
        ])
    else:
        transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(lambda x: x * 2 - 1) 
        ])

    return transforms

def save_visualization_outputs(predicted_images, save_path, epoch, rows=3, cols=4):
    """
    Use to save image results after each training epoch and validation epoch
    """
    plt.ioff()
    fig = plt.figure(figsize=(18, 12))
    for idx, img in enumerate(predicted_images, 1):
        plt.subplot(rows, cols, idx)
        plt.imshow((img.detach().numpy().transpose((1, 2, 0)) + 1) / 2)
        plt.axis("off")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f"{save_path}/{epoch}.jpg")
    plt.clf()
    plt.close(fig)


def plot_losses(args, history):
    train_discriminator_losses = [epoch_dict.get("train_discriminator_loss") for epoch_dict in history]
    train_generator_losses = [epoch_dict.get("train_generator_loss") for epoch_dict in history]
    plt.figure(figsize=(14, 8))
    plt.plot(train_discriminator_losses, color="r", label=" train_discriminator_loss")
    plt.plot(train_generator_losses, color="b", label="train_generator_losses")
    plt.title("Train losses over epochs")
    plt.legend()
    plt.savefig(f"{args.save_path}/train_losses.jpg")

    train_D_x = [epoch_dict.get("train_D_x") for epoch_dict in history]
    train_D_G_x = [epoch_dict.get("train_D_G_x") for epoch_dict in history]
    plt.figure(figsize=(14, 8))
    plt.plot(train_D_x, color="r", label=" train_D_x")
    plt.plot(train_D_G_x, color="b", label="train_D_G_x")
    plt.title("Train D_x/D_G_x over epochs")
    plt.legend()
    plt.savefig(f"{args.save_path}/train_d_x.jpg")

    val_discriminator_losses = [epoch_dict.get("val_discriminator_loss") for epoch_dict in history]
    val_generator_losses = [epoch_dict.get("val_generator_loss") for epoch_dict in history]
    plt.figure(figsize=(14, 8))
    plt.plot(val_discriminator_losses, color="r", label="val_discriminator_loss")
    plt.plot(val_generator_losses, color="b", label="val_generator_loss")
    plt.title("Validation losses over epochs")
    plt.legend()
    plt.savefig(f"{args.save_path}/Validation_losses.jpg")

    val_D_x = [epoch_dict.get("val_D_x") for epoch_dict in history]
    val_D_G_x = [epoch_dict.get("val_D_G_x") for epoch_dict in history]
    plt.figure(figsize=(14, 8))
    plt.plot(val_D_x, color="r", label=" val_D_x")
    plt.plot(val_D_G_x, color="b", label="val_D_G_x")
    plt.title("Validation D_x/D_G_x over epochs")
    plt.legend()
    plt.savefig(f"{args.save_path}/val_d_x.jpg")


def train(args, model, train_dataloader, val_dataloader, optimizers, device, model_saver):
    discriminator_optimizer, generator_optimizer = optimizers
    loss_fun = nn.BCELoss()
    history = []
    best_val_loss = math.inf
    visualization_train_path = os.path.join(args.save_path, "visualization_train")
    visualization_val_path = os.path.join(args.save_path, "visualization_val")

    real_labels = torch.ones(args.batch_size, 1, device=device)
    fake_labels = torch.zeros(args.batch_size, 1, device=device)

    for epoch in range(args.epochs):
        model.train()
        train_discriminator_loss, train_generator_loss = 0, 0
        train_D_x, train_D_G_x = 0, 0
        
        for real_images in tqdm(train_dataloader):
            bs = real_images.shape[0]
            real_images = real_images.to(device)
            fake_images = model.generator.generate(bs)
            
            discriminator_optimizer.zero_grad()
            out_fake = model.discriminator(fake_images.detach())
            discriminator_loss_fake = loss_fun(out_fake, fake_labels[:bs])

            out_real = model.discriminator(real_images)
            discriminator_loss_real = loss_fun(out_real, real_labels[:bs])
            train_D_x += out_real.sum()
            train_D_G_x += out_fake.sum()
            discriminator_loss = discriminator_loss_real + discriminator_loss_fake
            discriminator_loss.backward()
            discriminator_optimizer.step()
            train_discriminator_loss += discriminator_loss.item() * bs

            generator_optimizer.zero_grad()
            out_generator = model.discriminator(fake_images)
            generator_loss = loss_fun(out_generator, real_labels[:bs])
            train_D_G_x += out_generator.sum()
            generator_loss.backward()
            generator_optimizer.step()
            train_generator_loss += generator_loss.item() * bs
        save_visualization_outputs(fake_images[:10].cpu(), visualization_train_path, epoch)

        model.eval()
        val_discriminator_loss, val_generator_loss = 0, 0
        val_D_x, val_D_G_x = 0, 0
        with torch.no_grad():
            for real_images in tqdm(val_dataloader):
                bs = real_images.shape[0]
                real_images = real_images.to(device)
                fake_images = model.generator.generate(bs)
                
                out_fake = model.discriminator(fake_images)
                discriminator_loss_fake = loss_fun(out_fake, fake_labels[:bs])

                out_real = model.discriminator(real_images)
                discriminator_loss_real = loss_fun(out_real, real_labels[:bs])

                val_D_x += out_real.sum()
                val_D_G_x += out_fake.sum()
                discriminator_loss = discriminator_loss_real + discriminator_loss_fake
                val_discriminator_loss += discriminator_loss.item() * bs

                out_generator = model.discriminator(fake_images)
                generator_loss = loss_fun(out_generator, real_labels[:bs])
                val_D_G_x += out_generator.sum()
                val_generator_loss += generator_loss.item() * bs
        save_visualization_outputs(fake_images[:10].cpu(), visualization_val_path, epoch)
        
        train_discriminator_loss /= len(train_dataloader.sampler)
        train_generator_loss /= len(train_dataloader.sampler)
        train_D_x /= len(train_dataloader.sampler)
        train_D_G_x /= 2 * len(train_dataloader.sampler)
        val_discriminator_loss /= len(val_dataloader.sampler)
        val_generator_loss /= len(val_dataloader.sampler)
        val_D_x /= len(val_dataloader.sampler)
        val_D_G_x /= 2* len(val_dataloader.sampler)

        history.append({
            "epoch": epoch,
            "train_discriminator_loss": train_discriminator_loss,
            "train_generator_loss": train_generator_loss,
            "train_D_x": train_D_x.item(),
            "train_D_G_x": train_D_G_x.item(),
            "val_discriminator_loss": val_discriminator_loss,
            "val_generator_loss": val_generator_loss,
            "val_D_x": val_D_x.item(),
            "val_D_G_x": val_D_G_x.item(),
        })
        model_saver.save(model.state_dict(), val_generator_loss)
        best_val_loss = min(best_val_loss, val_generator_loss)
        if args.trial:
            args.trial.report(val_generator_loss, epoch) 
            if args.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        print('Epoch: {} Train Loss: {:.4f}, {:.4f}, {:.4f}, {:.4f} \
               Validation Loss: {:.4f}, {:.4f}, {:.4f}, {:.4f} '.format(epoch,
                                                                        train_generator_loss,
                                                                        train_discriminator_loss,
                                                                        train_D_x,
                                                                        train_D_G_x,
                                                                        train_generator_loss,
                                                                        train_discriminator_loss,
                                                                        val_D_x,
                                                                        val_D_G_x))
    return history, best_val_loss


def main(args):
    device = get_device(args.gpu)

    gan = GAN(3, args.noise_dimension, device).to(device)
    train_dataloader, val_dataloader, _ = create_dataloaders(args)
    discriminator_optimizer = create_optimizer(gan.discriminator, args)
    generator_optimizer = create_optimizer(gan.generator, args)
    model_saver = ModelSaver(args.save_path)

    history, best_val_loss = train(args, gan, train_dataloader, val_dataloader, (discriminator_optimizer, generator_optimizer), device, model_saver)
    plot_losses(args, history)

    return best_val_loss


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
