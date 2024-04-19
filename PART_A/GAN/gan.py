import torch
import torchvision
import os
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# loading mnist dataset
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
CHKPT_DIR = 'checkpoints'
os.makedirs(CHKPT_DIR, exist_ok=True)

# set params
batch_size=64
latent_sizes = [2, 4, 8, 16, 32, 64]
img_dim = 28*28*1 # single channel image
num_epochs = 10
lr = 2e-4

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Gelu(nn.Module):
    def __init__(self):
        super(Gelu, self).__init__()
        self.f = nn.functional.gelu
    def forward(self, x):
        return self.f(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, net_dim, img_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_dim = img_dim
        self.net_dim = net_dim
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, net_dim),
            Gelu(),
            nn.Linear(net_dim, img_dim),
            Gelu()
        )

    def forward(self, x):
        return self.generator(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels, net_dim):
        super(Discriminator, self).__init__()
        self.net_dim = net_dim
        self.discriminator = nn.Sequential(
            nn.Linear(in_channels, net_dim), Gelu(), nn.Linear(net_dim, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)

def make_models():
    generators = []
    for ls in latent_sizes:
        generators.append(Generator(ls, 128, img_dim))
    discriminator = Discriminator(img_dim, 128)
    return generators, discriminator

def train(dataloader=dataloader, num_epochs=num_epochs, lr=lr, batch_size=batch_size, img_dim=img_dim):
    generators, discriminator = make_models()
    criterion = nn.BCELoss()
    for generator in generators:
        print(f"Training generator with latent size {generator.latent_dim}")
        gen_optimizer = optim.Adam(generator.parameters(), lr=lr)
        disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr)
        for epoch in range(num_epochs):
            for i, (real_images, _) in enumerate(dataloader):
                real_images = real_images.view(-1, img_dim)
                real_labels = torch.ones(batch_size, 1)
                fake_labels = torch.zeros(batch_size, 1)

                # Backprop for discriminator
                z = torch.randn(batch_size, generator.latent_dim)
                fake_images = generator(z)
                real_outputs = discriminator(real_images)
                fake_outputs = discriminator(fake_images)
                d_loss_real = criterion(real_outputs, real_labels)
                d_loss_fake = criterion(fake_outputs, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                disc_optimizer.zero_grad()
                d_loss.backward()
                disc_optimizer.step()

                # Backprop for generator

                z = torch.randn(batch_size, generator.latent_dim)
                fake_images = generator(z)
                outputs = discriminator(fake_images)
                g_loss = criterion(outputs, real_labels)
                gen_optimizer.zero_grad()
                g_loss.backward()
                gen_optimizer.step()
                if i % 100 == 0:
                    print(f"Epoch {epoch}, iteration {i}, d_loss: {d_loss}, g_loss: {g_loss}")
        
        model_pth = os.path.join(CHKPT_DIR, f'generator_{generator.latent_dim}.pt')
        torch.save(generator.state_dict(), model_pth)
        print(f"Generator with Latent Size={generator.latent_dim} saved in: {model_pth}")

def visualise(generator):
    generator.eval()
    with torch.inference_mode():
        z = torch.randn(1, generator.latent_dim)
    return generator(z).view(-1, 28, 28)

if __name__ == '__main__':
    train()
    print(f"----------------Training Done! Checkpoints saved in {CHKPT_DIR}----------------")
    
    for i, latent_size in enumerate(latent_sizes):
        generator = Generator(latent_size, 128, img_dim)
        generator.load_state_dict(torch.load(os.path.join(CHKPT_DIR, f'generator_{latent_size}.pt')))
        torchvision.utils.save_image(visualise(generator), f'output_{i}.png')
    
    print(f"----------------Visualisation Done!----------------")
