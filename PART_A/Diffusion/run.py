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
from ddpm import DiffusionProcess, train_diffusion_model
from visualise import generate_samples, visualize_samples

transform = Compose(
    [ToTensor(), Lambda(lambda x: (x - 0.5) * 2)]
)
dataset = MNIST(root="./mnist_data", download=True, train=True, transform=transform)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dimensions = [8, 16, 32, 64]
num_epochs = 10
learning_rate = 0.001

trained_models = {}
for dim in latent_dimensions:
    print(f"Training with latent dimension: {dim}")
    trained_models[dim] = train_diffusion_model(
        base_channels=dim,
        data_loader=data_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
diffusion = DiffusionProcess()
for dim, model in trained_models.items():
    samples = generate_samples(model, diffusion, 16)
    visualize_samples(samples, f"fake_{dim}")
