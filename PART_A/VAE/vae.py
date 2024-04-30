import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os

os.makedirs("chkpts", exist_ok=True)

# MNIST Dataset
train_dataset = datasets.MNIST(
    root="./mnist_data/", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="./mnist_data/", train=False, transform=transforms.ToTensor(), download=False
)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=100, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=100, shuffle=False
)

class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)

        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

def loss_fn(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

latent_sizes = [2, 4, 8, 16, 32, 64]
epochs = 10

mses = {}

for latent_size in latent_sizes:
    print(f"Training with latent size: {latent_size}")
    
    vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=latent_size)
    
    if torch.cuda.is_available():
        vae.cuda()
        
    optimizer = optim.Adam(vae.parameters())
    
    for epoch in range(epochs):
        vae.train()
        train_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.cuda()
            optimizer.zero_grad()
            recon_batch, mu, log_var = vae(data)
            loss = loss_fn(recon_batch, data, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        
    checkpoint_path = f"chkpts/vae_{latent_size}.pt"
    torch.save(vae.state_dict(), checkpoint_path)
        
    vae.eval()
    test_loss = 0.0
    num_smpls=0
    
    criterion = nn.MSELoss(reduction='sum')
    
    with torch.inference_mode():
        for data, _ in test_loader:
            data = data.cuda()
            recon_batch, mu, log_var = vae(data)
            print(f"Mean: {mu.mean().item()}, Log Var: {log_var.mean().item()}")
            test_loss += criterion(recon_batch.view(-1,784), data.view(-1,784)).item()
            num_smpls+=len(data)
    mse = test_loss / num_smpls
    mses[latent_size] = mse
    
    print(f"MSE for latent_size: {latent_size}: {mse}")
