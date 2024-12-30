import os
import math
import optuna
import torch
from tqdm import tqdm
import torch.nn as nn
from argparse import ArgumentParser
from utils import ModelSaver, Visualizator, MetricsTracker, \
                  create_optimizer, create_dataloaders, get_device
from gan_model import GAN
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance


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
                    help="Size of the generator's noise input")

parser.add_argument("--batch-size", type=int, default=128,
                    help="training batch size")


def train(args, model, train_dataloader, val_dataloader, optimizers, device, model_saver, metricks_tracker, visualizator):
    discriminator_optimizer, generator_optimizer = optimizers
    mifid = MemorizationInformedFrechetInceptionDistance(reset_real_features=True, normalize=True).to(device)

    metricks_tracker.register_metric("generator_loss")
    metricks_tracker.register_metric("discriminator_loss")
    metricks_tracker.register_metric("D_x")
    metricks_tracker.register_metric("D_G_x")
    metricks_tracker.register_metric("mifid")

    for epoch in range(args.epochs):
        model.train()
        
        for real_images in tqdm(train_dataloader):
            bs = real_images.shape[0]
            real_images = real_images.to(device)
            fake_images = model.generator.generate(bs)

            real_labels = torch.FloatTensor(bs, 1).uniform_(0.9, 1.0).to(device)
            fake_labels = torch.FloatTensor(bs, 1).uniform_(0.0, 0.1).to(device)

            mifid.update((real_images + 1) / 2, real=True)
            mifid.update((fake_images + 1) / 2, real=False)
            
            discriminator_optimizer.zero_grad()
            out_fake = model.discriminator(fake_images.detach())
            out_real = model.discriminator(real_images)
            discriminator_loss = nn.BCELoss()(out_fake, fake_labels) + nn.BCELoss()(out_real, real_labels)
            discriminator_loss.backward()
            discriminator_optimizer.step()

            metricks_tracker.update_metric("D_x", out_real.sum().item(), bs, epoch, is_train=True)
            metricks_tracker.update_metric("D_G_x", out_fake.sum().item(), bs, epoch, is_train=True)
            metricks_tracker.update_metric("discriminator_loss", discriminator_loss.item() * bs, bs, epoch, is_train=True)

            generator_optimizer.zero_grad()
            out_generator = model.discriminator(fake_images)
            generator_loss = nn.BCELoss()(out_generator, real_labels)
            generator_loss.backward()
            generator_optimizer.step()
            
            metricks_tracker.update_metric("D_G_x", out_generator.sum().item(), bs, epoch, is_train=True)
            metricks_tracker.update_metric("generator_loss", generator_loss.item() * bs, bs, epoch, is_train=True)
        
        metricks_tracker.update_metric("mifid", mifid.compute().item(), 1, epoch, is_train=True)
        mifid.reset()
        visualizator.plot_images(fake_images[:10], epoch, is_train=True)

        model.eval()
        with torch.no_grad():
            for real_images in tqdm(val_dataloader):
                bs = real_images.shape[0]
                real_images = real_images.to(device)
                fake_images = model.generator.generate(bs)

                real_labels = torch.FloatTensor(bs, 1).uniform_(0.9, 1.0).to(device)
                fake_labels = torch.FloatTensor(bs, 1).uniform_(0.0, 0.1).to(device)

                mifid.update((real_images + 1) / 2, real=True)
                mifid.update((fake_images + 1) / 2, real=False)
                
                out_fake = model.discriminator(fake_images)
                out_real = model.discriminator(real_images)
                discriminator_loss = nn.BCELoss()(out_fake, fake_labels) + nn.BCELoss()(out_real, real_labels)

                metricks_tracker.update_metric("D_x", out_real.sum().item(), bs, epoch, is_train=False)
                metricks_tracker.update_metric("D_G_x", out_fake.sum().item(), bs, epoch, is_train=False)
                metricks_tracker.update_metric("discriminator_loss", discriminator_loss.item() * bs, bs, epoch, is_train=False)
                
                out_generator = model.discriminator(fake_images)
                generator_loss = nn.BCELoss()(out_generator, real_labels)

                metricks_tracker.update_metric("D_G_x", out_generator.sum().item(), bs, epoch, is_train=False)
                metricks_tracker.update_metric("generator_loss", generator_loss.item() * bs, bs, epoch, is_train=False)

        metricks_tracker.update_metric("mifid", mifid.compute().item(), 1, epoch, is_train=False)
        mifid.reset()
        visualizator.plot_images(fake_images[:10], epoch, is_train=False)
        
        val_mifid_value = metricks_tracker.get_metric("mifid", epoch, is_train=False)
        model_saver.save(model.state_dict(), val_mifid_value)

        if args.trial:
            args.trial.report(val_mifid_value, epoch) 
            if args.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        metricks_tracker.log_last_epoch()
    return metricks_tracker.get_best_value_of_metric("mifid", minimize=True, is_train=False)


def main(args):
    device = get_device(args.gpu)
    print(f"DEVICE NAME: {device}")

    gan = GAN(3, args.noise_dimension, device).to(device)
    train_dataloader, val_dataloader, _ = create_dataloaders(args, use_tanh=True)
    discriminator_optimizer = create_optimizer(gan.discriminator, args)
    generator_optimizer = create_optimizer(gan.generator, args)
    model_saver = ModelSaver(args.save_path)
    metricks_tracker = MetricsTracker()
    visualizator = Visualizator(args.save_path, use_tanh=True)

    best_val_mifid = train(args, gan, train_dataloader, val_dataloader, (discriminator_optimizer, generator_optimizer),\
                                    device, model_saver, metricks_tracker, visualizator)
    visualizator.plot_metrics(metricks_tracker)

    return best_val_mifid


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
