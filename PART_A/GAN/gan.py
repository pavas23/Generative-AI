import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

LATENT_SIZES = [2, 4, 8, 16, 32, 64]  # Gans are very sensitive to hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-4
BATCH_SIZE = 32
EPOCHS = 50

dataset = MNIST(root='data/', download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.3),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.generator(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.discriminator(x)

def train(latent_size):
    dis = Discriminator().to(device)
    gen = Generator(latent_size).to(device)
    f_noise = torch.randn((BATCH_SIZE, latent_size)).to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dis_opt = optim.Adam(dis.parameters(), lr=lr)
    gen_opt = optim.Adam(gen.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(EPOCHS):
        for batch_idx, (real, _) in enumerate(dataloader):
            
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]
            
            # Training Discriminator
            dis_opt.zero_grad()
            noise = torch.randn(batch_size, latent_size).to(device)
            fake = gen(noise)
            d_real = dis(real).view(-1)
            d_real_loss = criterion(d_real, torch.ones_like(d_real))
            d_fake = dis(fake).view(-1)
            d_fake_loss = criterion(d_fake, torch.zeros_like(d_fake))
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward(retain_graph=True)
            dis_opt.step()
            
            # Training Generator
            gen_opt.zero_grad()
            output = dis(fake).view(-1)
            g_loss = criterion(output, torch.ones_like(output))
            g_loss.backward()
            gen_opt.step()
            
            if batch_idx % 100 == 0:
                with torch.inference_mode():
                    fake = gen(f_noise)
                    fake = fake.view(-1, 1, 28, 28)
                    real = real.view(-1, 1, 28, 28)
                    fake_img_grid = torchvision.utils.make_grid(fake, normalize=True)
                
                    if epoch==EPOCHS-1:
                        torchvision.utils.save_image(fake_img_grid, f"fake_{latent_size}_{epoch+1}.png", normalize=True)
        print(f"Epoch: {epoch+1}, d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")
                    
for latent_size in LATENT_SIZES:
    print(f"Training with latent size: {latent_size}")
    train(latent_size)