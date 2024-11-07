import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, in_channels, embedding_size, device):
        super().__init__()

        hidden_channels = [32, 64, 128, 256, 512]
        self.encoder = Encoder(in_channels, hidden_channels, embedding_size, device).to(device)
        hidden_channels.reverse()
        self.decoder = Decoder(in_channels, hidden_channels, embedding_size, device).to(device)

    def forward(self, input):
        z, means, log_vars = self.encoder(input)
        out = self.decoder(z)
        return out, means, log_vars


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_size, device):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device

        self.layers = nn.ModuleList()

        hidden_channels = [in_channels] + hidden_channels
        for i in range(len(hidden_channels) - 1):
            self.layers.append(nn.Sequential(
                nn.Conv2d(hidden_channels[i], hidden_channels[i+1], kernel_size=(3,3), stride=2, padding=1),
                nn.BatchNorm2d(hidden_channels[i+1]),
                nn.LeakyReLU(),
            ))

        self.layers.append(nn.Flatten())
        
        self.block_mean = nn.Sequential(
            nn.Linear(2048, self.embedding_size),
        )

        self.block_log_var = nn.Sequential(
            nn.Linear(2048, self.embedding_size),
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
    def __init__(self, out_channels, hidden_channels, embedding_size, device):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = device

        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(
            nn.Linear(in_features=self.embedding_size, out_features=2048),
            nn.Unflatten(dim=1, unflattened_size=(512, 2, 2)),
        ))
        
        for i in range(len(hidden_channels) - 1):
            self.layers.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i+1], kernel_size=(3,3), stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_channels[i+1]),
                nn.LeakyReLU(),
            ))

        self.layers.append(nn.Sequential(
            nn.ConvTranspose2d(hidden_channels[-1], hidden_channels[-1], kernel_size=(3,3), stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_channels[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=hidden_channels[-1], out_channels=out_channels, kernel_size=(3,3), padding=1),
            nn.Sigmoid()
        ))
        
    def forward(self, input):
        out = input
        for layer in self.layers:
            out = layer(out)
        return out
  