import torch
import torch.nn as nn

class GAN(nn.Module):
    def __init__(self, in_channels, device):
        super().__init__()
        hidden_channels = [512, 256, 128, 64, 32, in_channels]
        self.generator = Generator(hidden_channels, device)
        hidden_channels.reverse()
        self.discriminator = Discriminatior(hidden_channels)


class Generator(nn.Module):
    def __init__(self, hidden_channels, device):
        super().__init__()
        self.layers = nn.ModuleList()
        self.device = device

        self.layers.append(nn.Linear(1024, 2048))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Unflatten(dim=1, unflattened_size=(512, 2, 2)))

        for i in range(len(hidden_channels) - 1):
            self.layers.append(self.create_generator_block(hidden_channels[i], hidden_channels[i + 1]))
        self.layers.append(nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels[-1], out_channels=hidden_channels[-1], kernel_size=(3,3), padding=1),
            nn.Sigmoid()
        ))

    def create_generator_block(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=(3,3), stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(output_channels)
        )

    def generate(self, batch_size):
        out = torch.randn(batch_size, 1024, device=self.device)
        for layer in self.layers:
            out = layer(out)
        return out


class Discriminatior(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(len(hidden_channels) - 1):
            self.layers.append(self.create_discriminator_block(hidden_channels[i], hidden_channels[i+1]))

        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(2048, 1024))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=0.2))
        self.layers.append(nn.Linear(1024, 1))

    def create_discriminator_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, input):
        out = input
        for layer in self.layers:
            out = layer(out)
        return out
