import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Lambda
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from unet import UNet

class DiffusionProcess:
    def __init__(self, beta_start=0.0001, beta_end=0.02, num_steps=1000):
        self.num_steps = num_steps
        self.beta = torch.linspace(beta_start, beta_end, num_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        self.beta_cumprod = 1.0 - self.alpha_cumprod

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])
        sqrt_beta_cumprod = torch.sqrt(self.beta_cumprod[t])
        return (
            sqrt_alpha_cumprod.view(-1, 1, 1, 1) * x_start
            + sqrt_beta_cumprod.view(-1, 1, 1, 1) * noise
        )

    def sample_t(self, num_samples):
        return torch.randint(0, self.num_steps, (num_samples,))

    def p_mean_variance(self, model, x, t):
        noise_prediction = model(x, t)
        mean, variance = self.q_mean_variance(x_start=None, t=t)
        predicted_mean = mean - variance * noise_prediction
        return predicted_mean, variance


def train_diffusion_model(base_channels, data_loader, num_epochs, learning_rate):
    model = UNet(in_channels=1, base_channels=base_channels)
    diffusion = DiffusionProcess()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, _ in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
            images = images

            # Generate timesteps and add noise to images
            timesteps = diffusion.sample_t(len(images))
            noise = torch.randn_like(images)
            noisy_images = diffusion.q_sample(images, timesteps, noise=noise)

            # Predict noise using the UNet model (single argument)
            predicted_noise = model(noisy_images)  # Only one argument

            # Calculate loss and update the model
            loss = loss_fn(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss {epoch_loss:.4f}")

    return model
