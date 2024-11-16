import os
import json
import optuna
from datetime import datetime
from argparse import ArgumentParser
from functools import partial
from dotmap import DotMap
from train_vae_model import main as train_model


parser = ArgumentParser("Arguments for searching optimal hyperparameters")

parser.add_argument("--n-trials", "--n", type=int, default=100,
                    help="Number of times the model will we trained with different hyperparameters")

parser.add_argument("--dataset-path", type=str, required=True,
                    help="Path to the directory where train, val, test sets are stored")

parser.add_argument("--gpu", type=int, default=1,
                    help="1 - use gpu, 0 - use cpu")

parser.add_argument("--epochs", type=int, default=100,
                    help="Number of training epochs for each run")


def sample_hyperparameters(trial):
    args_dot_map = DotMap()
    args_dot_map.lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    args_dot_map.embedding_size = trial.suggest_categorical("embed_size", [64, 128, 256, 512, 1024])
    args_dot_map.optimizer = trial.suggest_categorical("optim", ["Adam", "AdamW", "SGD"])
    args_dot_map.weight_decay = trial.suggest_float("decay", 0, 0.3)
    args_dot_map.momentum = trial.suggest_float("momentum", 0.8, 0.99)
    args_dot_map.batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    return args_dot_map


def update_dot_map_with_args(args_dot_map, trial, args):
    args_dot_map.dataset_path = args.dataset_path
    args_dot_map.save_path = f"optuna_search/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    args_dot_map.gpu = args.gpu
    args_dot_map.epochs = args.epochs
    args_dot_map.trial = trial
    args_dot_map.trial_number = trial.number
    return args_dot_map


def serializable_dict(args_dot_map):
    return {"lr": args_dot_map.lr,
            "embedding_size": args_dot_map.embedding_size,
            "optimizer": args_dot_map.optimizer,
            "weight_decay": args_dot_map.weight_decay,
            "momentum": args_dot_map.momentum,
            "batch_size": args_dot_map.batch_size,
            "dataset_path": args_dot_map.dataset_path,
            "save_path": args_dot_map.save_path,
            "gpu": args_dot_map.gpu,
            "epochs": args_dot_map.epochs,
            "trial_number": args_dot_map.trial_number,
            }
            


def save_hyperparameters_logs(args_dot_map):
    file_path = os.path.join(args_dot_map.save_path, "hyperparameters")
    os.makedirs(args_dot_map.save_path, exist_ok=True)
    with open(file_path, "w") as outfile: 
        json.dump(serializable_dict(args_dot_map), outfile)


def objective_fn(trial, args):
    args_dot_map = sample_hyperparameters(trial)
    args_dot_map = update_dot_map_with_args(args_dot_map, trial, args)
    save_hyperparameters_logs(args_dot_map)
    loss = train_model(args_dot_map)
    return loss


def main(args):
    study = optuna.create_study(direction="minimize")
    objective = partial(objective_fn, args=args)
    study.optimize(objective, n_trials=args.n_trials)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print(f" No. All trials: {len(study.trials)}")
    print(f" No. Finished trials: {len(complete_trials)}")
    print(f" No. Pruned trials: {len(pruned_trials)}")

    print(f"Best trial value: {study.best_trial.value}")
    for key, value in study.best_trial.params.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
