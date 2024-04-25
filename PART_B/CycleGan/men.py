import torch
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, image_dir, annotations, transform=None):
        self.image_dir = image_dir
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_name = self.annotations.iloc[idx]["image_id"]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image

dataset_dir = "/mnt/MIG_store/Datasets/celeba/img_align_celeba/img_align_celeba"
annotations = pd.read_csv("/mnt/MIG_store/Datasets/celeba/list_attr_celeba.csv")

def get_subset(attr_data, attr_name, positive=True):
    if positive:
        indices = np.where(attr_data[attr_name] == 1)[0]
    else:
        indices = np.where(attr_data[attr_name] == -1)[0]
    return indices


men_without_glasses_indices = np.intersect1d(
    get_subset(annotations, "Male", positive=True),
    get_subset(annotations, "Eyeglasses", positive=False),
)

men_with_glasses_indices = np.intersect1d(
    get_subset(annotations, "Male", positive=True),
    get_subset(annotations, "Eyeglasses", positive=True),
)

women_with_glasses_indices = np.intersect1d(
    get_subset(annotations, "Male", positive=False),
    get_subset(annotations, "Eyeglasses", positive=True),
)

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

celeba_dataset = CustomDataset(dataset_dir, annotations, transform=transform)

men_without_glasses = Subset(celeba_dataset, men_without_glasses_indices)
men_with_glasses = Subset(celeba_dataset, men_with_glasses_indices)
women_with_glasses = Subset(celeba_dataset, women_with_glasses_indices)

batch_size = 16  # Desired batch size
dataloader1 = DataLoader(men_without_glasses, batch_size=batch_size, shuffle=True)
dataloader2 = DataLoader(men_with_glasses, batch_size=batch_size, shuffle=True)
dataloader3 = DataLoader(women_with_glasses, batch_size=batch_size, shuffle=True)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_residual_blocks=9):
        super(Generator, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.residuals = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual_blocks)]
        )

        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.down(x)
        x = self.residuals(x)
        x = self.up(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
        )

    def forward(self, x):
        return self.model(x)

lr = 0.0002
beta1 = 0.5

generator_x2y = Generator()
generator_y2x = Generator()
discriminator_x = Discriminator()
discriminator_y = Discriminator()

optimizer_G = optim.Adam(
    list(generator_x2y.parameters()) + list(generator_y2x.parameters()),
    lr=lr,
    betas=(beta1, 0.999),
)

optimizer_dx = optim.Adam(discriminator_x.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_dy = optim.Adam(discriminator_y.parameters(), lr=lr, betas=(beta1, 0.999))

criterion_gan = nn.MSELoss()
criterion_cycle = nn.L1Loss()

num_epochs=50
lambda_cycle = 10 # weight for cycle consistency

for epoch in range(num_epochs):
    print(f"...Epoch started...")
    total_loss = 0.
    for(batch_X, batch_Y) in zip(dataloader1, dataloader2):
        real_X = batch_X
        real_Y = batch_Y

        fake_Y = generator_x2y(real_X)
        pred_fake_Y = discriminator_y(fake_Y)
        gan_loss_XtoY = criterion_gan(pred_fake_Y, torch.ones_like(pred_fake_Y))

        fake_X = generator_y2x(real_Y)
        pred_fake_X = discriminator_x(fake_X)
        gan_loss_YtoX = criterion_gan(pred_fake_X, torch.ones_like(pred_fake_X))

        recovered_X = generator_y2x(fake_Y)
        cycle_loss_X = criterion_cycle(recovered_X, real_X)

        recovered_Y = generator_x2y(fake_X)
        cycle_loss_Y = criterion_cycle(recovered_Y, real_Y)

        total_G_loss = gan_loss_XtoY + gan_loss_YtoX + lambda_cycle * (cycle_loss_X + cycle_loss_Y)

        optimizer_G.zero_grad()
        total_G_loss.backward()
        optimizer_G.step()
        total_loss += total_G_loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] : Loss : {total_loss}")


def display_images(images, title=""):
    _, ax = plt.subplots(1, len(images), figsize=(15, 5))
    for idx, img in enumerate(images):
        img = img.permute(1, 2, 0).detach().cpu() 
        img = img * 0.5 + 0.5  
        ax[idx].imshow(img)
        ax[idx].axis("off")
    plt.suptitle(title)
    plt.savefig(f"{title}.png")


test_images_X = [real_X[0] for real_X, _ in dataloader1][:5] 
test_images_Y = [real_Y[0] for real_Y, _ in dataloader2][:5] 

fake_Y_images = [
    generator_x2y(img.unsqueeze(0)) for img in test_images_X
] 
fake_X_images = [
    generator_y2x(img.unsqueeze(0)) for img in test_images_Y
]  

display_images(test_images_X, "Original Men without Glasses")
display_images(fake_Y_images, "Translated to Men with Glasses")
display_images(test_images_Y, "Original Men with Glasses")
display_images(fake_X_images, "Translated to Men without Glasses")
