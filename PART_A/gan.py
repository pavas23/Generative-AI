import torch
import torchvision
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# loading mnist dataset
dataset = Dataset(datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])))

# set params
batch_size=64
latent_sizes = [2, 4, 8, 16, 32, 64]
img_dim = 28*28*1 # single channel image
num_epochs = 100
lr = 2e-4

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, latent_dim, net_dim, img_dim):
        self.generator = nn.Sequential(
            [
                nn.Linear(latent_dim, net_dim),
                nn.GELU(),
                nn.Linear(net_dim, img_dim),
                nn.GeLU()
            ]
        )
        
    def forward(self, x):
        return self.generator(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, net_dim):
        self.discriminator = nn.Sequential(
            [
                nn.Linear(in_channels, net_dim),
                nn.GELU(),
                nn.Linear(net_dim, 1),
                nn.Sigmoid()
            ]
        )
    
    def forward(self, x):
        return self.discriminator(x)

def make_models():
    generators = []
    for ls in latent_sizes:
        generators.append(Generator(ls, 128, img_dim))
    discriminator = Discriminator(img_dim, 128)
    return generators, discriminator

def train(generators, discriminator, dataloader, num_epochs, lr):
    generators, discriminator = make_models()
    criterion = nn.BCELoss()