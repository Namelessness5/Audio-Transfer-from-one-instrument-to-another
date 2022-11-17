import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class transfer(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size, depth=2, method="linear", conv_dim=128, kernel_size=3, bias=False):
        super(transfer, self).__init__()

        layers = []
        if method == "linear":
            for i in range(depth):
                layers.append(nn.Linear(input_size, input_size, bias=bias))
                layers.append(nn.LeakyReLU(0.01))
        elif method == "conv":
            layers.append(nn.Conv1d(1, conv_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias))
            for i in range(depth - 2):
                layers.append(nn.Conv1d(conv_dim, conv_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias))
                layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Conv1d(conv_dim, 1, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias))

        self.mainC = nn.Sequential(*layers)

    def forward(self, x):
        return self.mainC(x)