import os
import math
import torch


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
        self.T = num_epochs * iterations_for_epoch
        self.M = number_of_cycles
        self.R = R

        self.f_a = 1 / self.R
        self.f_b = 0

    def get_updated_beta(self, iteration):
        tau = ((iteration - 1) % math.ceil(self.T / self.M)) / (self.T / self.M)
        if tau > self.R:
            return 1
        return self.f_a * tau + self.f_b