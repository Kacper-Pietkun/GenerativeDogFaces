import torch
import optuna
import torch.nn as nn
from utils import ModelSaver, BetaCyclicalAnnealing, Visualizator, MetricsTracker, \
                  create_optimizer, create_dataloaders, get_device
from argparse import ArgumentParser
from vae_model import VAE
from tqdm import tqdm
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance


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


def train(args, model, train_dataloader, val_dataloader, optimizer, device, beta_cyclical_annealing, model_saver, metricks_tracker, visualizator):
    mifid = MemorizationInformedFrechetInceptionDistance(reset_real_features=True, normalize=True).to(device)

    metricks_tracker.register_metric("loss")
    metricks_tracker.register_metric("rec_loss")
    metricks_tracker.register_metric("kl_loss")

    for epoch in range(args.epochs):
        model.train()

        for input_image in tqdm(train_dataloader):
            bs = input_image.shape[0]
            input_image = input_image.to(device)

            optimizer.zero_grad()
            predicted_image, mean, log_var = model(input_image)
            curr_beta = beta_cyclical_annealing.get_updated_beta()
            loss, rec_loss, kl_loss = loss_fn(input_image, predicted_image, mean, log_var, curr_beta)
            loss.backward()
            optimizer.step()

            metricks_tracker.update_metric("loss", loss.item() * bs, bs, epoch, is_train=True)
            metricks_tracker.update_metric("rec_loss", rec_loss.item() * bs, bs, epoch, is_train=True)
            metricks_tracker.update_metric("kl_loss", kl_loss.item() * bs, bs, epoch, is_train=True)
        
        visualizator.plot_images(predicted_image[:10], epoch, is_train=True)

        model.eval()
        with torch.no_grad():
            for input_image in tqdm(val_dataloader):
                bs = input_image.shape[0]
                input_image = input_image.to(device)

                predicted_image, mean, log_var = model(input_image)
                loss, rec_loss, kl_loss = loss_fn(input_image, predicted_image, mean, log_var, 1)

                metricks_tracker.update_metric("loss", loss.item() * bs, bs, epoch, is_train=False)
                metricks_tracker.update_metric("rec_loss", rec_loss.item() * bs, bs, epoch, is_train=False)
                metricks_tracker.update_metric("kl_loss", kl_loss.item() * bs, bs, epoch, is_train=False)

        visualizator.plot_images(predicted_image[:10], epoch, is_train=False)
        
        val_loss = metricks_tracker.get_metric("loss", epoch, is_train=False)
        model_saver.save(model.state_dict(), val_loss)

        if args.trial:
            args.trial.report(val_loss, epoch) 
            if args.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
        metricks_tracker.log_last_epoch()
    return metricks_tracker.get_best_value_of_metric("loss", minimize=True, is_train=False)


def main(args):
    device = get_device(args.gpu)
    print(f"DEVICE NAME: {device}")

    model = VAE(3, args.embedding_size, device).to(device)
    train_dataloader, val_dataloader, _ = create_dataloaders(args, use_tanh=False)
    optimizer = create_optimizer(model, args)
    model_saver = ModelSaver(args.save_path)
    metricks_tracker = MetricsTracker()
    visualizator = Visualizator(args.save_path, use_tanh=False)
    beta_cyclical_annealing = BetaCyclicalAnnealing(args.epochs, len(train_dataloader))
    
    best_val_loss = train(args, model, train_dataloader, val_dataloader, optimizer, device, beta_cyclical_annealing, \
                           model_saver, metricks_tracker, visualizator)
    visualizator.plot_metrics(metricks_tracker)

    return best_val_loss


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
