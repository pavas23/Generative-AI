import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from unet import UNet

class DiffusionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.unet = UNet(1, 1)
        self.loss_fn = nn.MSELoss()
        self.num_timesteps = 1000
        self.betas = torch.linspace(-20, 20, self.num_timesteps)

    def forward(self, x, t):
        noise = torch.randn_like(x)
        alpha = torch.cos(self.betas[t]).pow(2)
        alpha_bar = torch.sin(self.betas[t]).pow(2)
        x_noisy = (x * torch.sqrt(alpha)) + (noise * torch.sqrt(alpha_bar))

        output = self.unet(x_noisy)

        return output

    def training_step(self, batch, batch_idx):
        images, _ = batch
        t = torch.randint(0, self.num_timesteps, (images.size(0),))
        preds = self(images, t)
        loss = self.loss_fn(preds, images)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def train_dataloader(self):
        dataset = MNIST(
            "data", train=True, download=True, transform=transforms.ToTensor()
        )
        return DataLoader(dataset, batch_size=64, shuffle=True)
