import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, embedding_size, device):
        super().__init__()

        self.encoder = Encoder(embedding_size, device).to(device)
        self.decoder = Decoder(embedding_size, device).to(device)

    def forward(self, input):
        z, means, log_vars = self.encoder(input)
        out = self.decoder(z)
        return out, means, log_vars


class Encoder(nn.Module):
    def __init__(self, embedding_size, device):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device

        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Dropout(0.5)
        ))
        
        self.layers.append(nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Dropout(0.5)
        ))
        
        self.layers.append(nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.5)
        ))
        
        self.block_mean = nn.Sequential(
            nn.Linear(6272, self.embedding_size),
        )

        self.block_log_var = nn.Sequential(
            nn.Linear(6272, self.embedding_size),
        )


    def forward(self, input):
        z = input
        for layer in self.layers:
            z = layer(z)

        means = self.block_mean(z)
        log_vars = self.block_log_var(z)
        standard_normal_dist = torch.distributions.Normal(torch.zeros_like(means), torch.ones_like(log_vars))
        eps = standard_normal_dist.sample().to(self.device)
        z = means + torch.exp(0.5 * log_vars) * eps
        return z, means, log_vars
  

class Decoder(nn.Module):
    def __init__(self, embedding_size, device):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device

        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(
            nn.Linear(in_features=self.embedding_size, out_features=256*4*4),
            nn.ReLU(),
        ))

        self.layers.append(nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(256, 4, 4)),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ))

        self.layers.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ))
        
        self.layers.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        ))
        
        self.layers.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        ))
        
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1),
            nn.Sigmoid()
        ))
        
    def forward(self, input):
        out = input
        for layer in self.layers:
            out = layer(out)
        return out
  