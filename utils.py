import os
import math
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset

class ModelSaver():
    """
    Class responsible for saving the best model during training.
    The best model is considered the one with the lowest loss function
    """
    def __init__(self, save_path, best_loss=math.inf):
        self.best_loss = best_loss
        self.save_path = os.path.join(save_path, "best_model.pt")

    def save(self, model_state_dict, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(model_state_dict, self.save_path)
            print("************** saved new best model **************")


class BetaCyclicalAnnealing():
    """
    Beta value scheduler used for loss calculation 
    (beta determines how much KL divergance factor affects loss function)
    """
    def __init__(self, num_epochs, iterations_for_epoch, number_of_cycles=4, R=0.5):
        self.iteration = 0
        self.T = num_epochs * iterations_for_epoch
        self.M = number_of_cycles
        self.R = R

        self.f_a = 1 / self.R
        self.f_b = 0

    def get_updated_beta(self):
        self.iteration += 1
        tau = ((self.iteration - 1) % math.ceil(self.T / self.M)) / (self.T / self.M)
        if tau > self.R:
            return 1
        return self.f_a * tau + self.f_b
    

class TensorDatasetWithAugmentations(Dataset):
    def __init__(self, tensor, transforms):
        self.tensor = tensor
        self.transforms = transforms
    
    def __getitem__(self, index):
        return self.transforms(self.tensor[index])
    
    def __len__(self):
        return self.tensor.shape[0]


class MetricsTracker:

    def __init__(self):
        self.metrics = {}
        self.index = 0

    class Metric:
        def __init__(self, name, is_plotable):
            self.name = name
            self.is_plotable = is_plotable
            self.train_values, self.train_sizes = [], []
            self.val_values, self.val_sizes = [], []

    def __iter__(self):
        self.index = 0
        return self
    
    def __next__(self):
        if self.index >= len(self.metrics):
            raise StopIteration
        metric = list(self.metrics.values())[self.index]
        self.index += 1
        return (metric.name, 
                [a / b for a, b in zip(metric.train_values, metric.train_sizes)],
                [a / b for a, b in zip(metric.val_values, metric.val_sizes)],
                metric.is_plotable)

    def register_metric(self, metric_name, is_plotable=True):
        if metric_name not in self.metrics.keys():
            new_metric = self.Metric(metric_name, is_plotable)
            self.metrics[metric_name] = new_metric

    def update_metric(self, metric_name, value, curr_bs, epoch, is_train):
        if metric_name not in self.metrics.keys():
            raise AssertionError(f"Metric: '{metric_name}' was not registered")
        
        metric = self.metrics[metric_name]
        target_values = metric.train_values if is_train else metric.val_values
        target_sizes = metric.train_sizes if is_train else metric.val_sizes

        if epoch > len(target_values) + 1:
            raise ValueError("Metric values should be reported every epoch.")
        
        if epoch == len(target_sizes):
            target_values.append(0)
            target_sizes.append(0)

        target_values[epoch] += value
        target_sizes[epoch] += curr_bs

    def get_metric(self, metric_name, is_train):
        if metric_name not in self.metrics.keys():
            raise AssertionError(f"Metric: '{metric_name}' was not registered")
        
        metric = self.metrics[metric_name]
        target_values = metric.train_values if is_train else metric.val_values
        target_sizes = metric.train_sizes if is_train else metric.val_sizes
        
        return [a / b for a, b in zip(target_values, target_sizes)]
    

    def get_metric(self, metric_name, epoch, is_train):
        if metric_name not in self.metrics.keys():
            raise AssertionError(f"Metric: '{metric_name}' was not registered")
        
        metric = self.metrics[metric_name]
        target_values = metric.train_values if is_train else metric.val_values
        target_sizes = metric.train_sizes if is_train else metric.val_sizes

        return target_values[epoch] / target_sizes[epoch]
    
    def get_best_value_of_metric(self, metric_name, minimize, is_train):
        if metric_name not in self.metrics.keys():
            raise AssertionError(f"Metric: '{metric_name}' was not registered")
        
        metric = self.metrics[metric_name]
        target_values = metric.train_values if is_train else metric.val_values
        target_sizes = metric.train_sizes if is_train else metric.val_sizes

        if minimize:
            return min([a / b for a, b in zip(target_values, target_sizes)])
        return max([a / b for a, b in zip(target_values, target_sizes)])
    
    def log_last_epoch(self):
        metric_names = []
        train_values = []
        val_values = []

        for name, metric in self.metrics.items():
            metric_names.append(name)
            train_values.append(metric.train_values[-1] / metric.train_sizes[-1])
            val_values.append(metric.val_values[-1] / metric.val_sizes[-1])

        train_values = [str(round(x, 3)) for x in train_values]
        val_values = [str(round(x, 3)) for x in val_values]
        epoch = len(next(iter(self.metrics.values())).train_values) - 1
        msg = f"Epoch: {epoch} | ({', '.join(metric_names)}) | Train: {', '.join(train_values)} | Val: {', '.join(val_values)}"
        print(msg)


class Visualizator:
    def __init__(self, save_path, use_tanh=False):
        self.base_save_path = save_path
        self.train_path = os.path.join(save_path, "visualization_train")
        self.val_path = os.path.join(save_path, "visualization_val")
        self.use_tanh = use_tanh

    def plot_images(self, predicted_images, epoch, is_train, rows=3, cols=4):
        save_path = self.train_path if is_train else self.val_path
        predicted_images = predicted_images.cpu()
        plt.ioff()
        fig = plt.figure(figsize=(18, 12))
        for idx, img in enumerate(predicted_images, 1):
            plt.subplot(rows, cols, idx)
            img = img.detach().numpy().transpose((1, 2, 0))
            if self.use_tanh:
                img = (img + 1) / 2
            plt.imshow(img)
            plt.axis("off")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(f"{save_path}/{epoch}.jpg")
        plt.clf()
        plt.close(fig)

    def plot_metrics(self, metrics_tracker):
        for name, train_values, val_values, is_plotable in metrics_tracker:
            if not is_plotable:
                continue

            plt.figure(figsize=(14, 8))
            plt.plot(train_values, color="r", label=f"train_{name}")
            plt.plot(val_values, color="b", label=f"validation_{name}")
            plt.title(f"Train and validation {name} over epochs")
            plt.legend()
            plt.savefig(f"{self.base_save_path}/train_val_{name}.jpg")


class Augmentator:
    def __init__(self, use_tanh=False):
        train_transforms = [
            v2.ToDtype(torch.uint8, scale=True),
            v2.RandomAdjustSharpness(sharpness_factor=3, p=0.5),
            v2.RandomEqualize(p=0.5),
            v2.RandomAutocontrast(0.5),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True)]
        
        test_transforms = [
            v2.ToDtype(torch.float32, scale=True),
        ]

        if use_tanh:
            train_transforms.append(v2.Lambda(lambda x: x * 2 - 1))
            test_transforms.append(v2.Lambda(lambda x: x * 2 - 1))

        self.train_augs = v2.Compose(train_transforms)
        self.test_augs = v2.Compose(test_transforms)


def get_device(use_gpu):
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:0")
        else:
            raise ValueError("GPU training was chosen but cuda is not available")
    else:
        device = torch.device("cpu")
    return device


def create_dataloaders(args, use_tanh=True):
    augmentator = Augmentator(use_tanh=use_tanh)

    train_set = np.load(os.path.join(args.dataset_path, "train_images.npy"))
    train_dataset = TensorDatasetWithAugmentations(train_set, augmentator.train_augs)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    validation_set = np.load(os.path.join(args.dataset_path, "validation_images.npy"))
    validation_dataset = TensorDatasetWithAugmentations(validation_set, augmentator.test_augs)
    val_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    test_set = np.load(os.path.join(args.dataset_path, "test_images.npy"))
    test_dataset = TensorDatasetWithAugmentations(test_set, augmentator.test_augs)
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
