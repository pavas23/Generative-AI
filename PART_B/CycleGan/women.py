import torch
import torchvision
from itertools import chain
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

transform = transforms.Compose(
    [
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)

batch_size = 32
lr = 0.0002
epochs = 10


class Generator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.final_conv = nn.Conv2d(512, 1, 4, 1, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_conv(x)
        return x


class CelebADataset(Dataset):
    def __init__(self, data_dir, mode, transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        self.df = pd.read_csv(os.path.join(data_dir, "list_attr_celeba.csv"))

        if mode == "men_no_glasses":
            self.df = self.df[(self.df["Male"] == 1) & (self.df["Eyeglasses"] == -1)]
        elif mode == "men_with_glasses":
            self.df = self.df[(self.df["Male"] == 1) & (self.df["Eyeglasses"] == 1)]
        elif mode == "women_with_glasses":
            self.df = self.df[(self.df["Male"] == -1) & (self.df["Eyeglasses"] == 1)]
        else:
            raise ValueError("Invalid mode")

        self.image_paths = [
            os.path.join(data_dir, "img_align_celeba", "img_align_celeba", f"{img_id}")
            for img_id in self.df["image_id"]
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image


men_no_glasses_dataset = CelebADataset(
    "/mnt/MIG_store/Datasets/celeba", "men_no_glasses", transform=transform
)
men_with_glasses_dataset = CelebADataset(
    "/mnt/MIG_store/Datasets/celeba", "men_with_glasses", transform=transform
)
women_with_glasses_dataset = CelebADataset(
    "/mnt/MIG_store/Datasets/celeba", "women_with_glasses", transform=transform
)

men_no_glasses_loader = DataLoader(
    men_no_glasses_dataset, batch_size=batch_size, shuffle=True
)
men_with_glasses_loader = DataLoader(
    men_with_glasses_dataset, batch_size=batch_size, shuffle=True
)
women_with_glasses_loader = DataLoader(
    women_with_glasses_dataset, batch_size=batch_size, shuffle=True
)


class CycleGAN(nn.Module):
    def __init__(self, generator_A2B, generator_B2A, discriminator_A, discriminator_B):
        super(CycleGAN, self).__init__()
        self.gen_A2B = generator_A2B
        self.gen_B2A = generator_B2A
        self.disc_A = discriminator_A
        self.disc_B = discriminator_B

    def forward(self, real_A, real_B):
        fake_B = self.gen_A2B(real_A)
        fake_A = self.gen_B2A(real_B)

        cycle_A = self.gen_B2A(fake_B)
        cycle_B = self.gen_A2B(fake_A)

        disc_real_A = self.disc_A(real_A)
        disc_real_B = self.disc_B(real_B)
        disc_fake_A = self.disc_A(fake_A.detach())
        disc_fake_B = self.disc_B(fake_B.detach())

        return (
            fake_A,
            fake_B,
            cycle_A,
            cycle_B,
            disc_real_A,
            disc_real_B,
            disc_fake_A,
            disc_fake_B,
        )


def loss(disc_real, disc_fake):
    real_loss = F.mse_loss(disc_real, torch.ones_like(disc_real))
    fake_loss = F.mse_loss(disc_fake, torch.zeros_like(disc_fake))
    return (real_loss + fake_loss) / 2


def cycle_loss(real, cycle):
    return F.l1_loss(real, cycle)


gen_A2B = Generator(input_channels=3, output_channels=3)
gen_B2A = Generator(input_channels=3, output_channels=3)
disc_A = Discriminator(input_channels=3)
disc_B = Discriminator(input_channels=3)

cycle_gan = CycleGAN(gen_A2B, gen_B2A, disc_A, disc_B)

gen_opt = optim.Adam(
    chain(gen_A2B.parameters(), gen_B2A.parameters()),
    lr=lr,
    betas=(0.5, 0.999),
)
disc_opt = optim.Adam(
    chain(disc_A.parameters(), disc_B.parameters()), lr=lr, betas=(0.5, 0.999)
)

for epoch in range(epochs):
    total_loss = 0.0
    for real_A, real_B in zip(women_with_glasses_loader, men_with_glasses_loader):
        (
            fake_A,
            fake_B,
            cycle_A,
            cycle_B,
            disc_real_A,
            disc_real_B,
            disc_fake_A,
            disc_fake_B,
        ) = cycle_gan(real_A, real_B)

        gen_loss = (
            loss(disc_fake_A, disc_fake_B)
            + cycle_loss(real_A, cycle_A)
            + cycle_loss(real_B, cycle_B)
        )
        disc_loss = loss(disc_real_A, disc_fake_A) + loss(disc_real_B, disc_fake_B)

        gen_opt.zero_grad()
        gen_loss.backward(retain_graph=True)
        gen_opt.step()

        disc_opt.zero_grad()
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        total_loss = gen_loss + disc_loss
    print(f"Epoch [{epoch}/{epochs}] Loss: {total_loss.item()}")

sample_real_A = next(iter(men_no_glasses_loader))[:5]
sample_fake_B = gen_A2B(sample_real_A)
torchvision.utils.save_image(
    sample_fake_B, "men_no_glasses_to_men_with_glasses.png", nrow=5, normalize=True
)

sample_real_B = next(iter(men_with_glasses_loader))[:5]
sample_fake_A = gen_B2A(sample_real_B)
torchvision.utils.save_image(
    sample_fake_A, "men_with_glasses_to_men_no_glasses.png", nrow=5, normalize=True
)
