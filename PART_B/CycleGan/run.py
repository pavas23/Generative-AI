import torch, os
import torch.nn as nn
import itertools
from dataloaders import men_no_glasses_loader, men_with_glasses_loader, women_with_glasses_loader

os.makedirs("checkpoints", exist_ok=True)

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
        return self.decoder(self.encoder(x))

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
        
        self.linear = nn.Linear(512*15*15,1000)
        self.linear2 = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear(x.view(-1, 512*15*15))
        x = self.linear2(x)
        return torch.sigmoid(x)

class CycleGAN(nn.Module):
    def __init__(
        self,
        generator_A2B,
        generator_B2A,
        discriminator_A,
        discriminator_B,
        lambda_cycle=10.0,
    ):
        super(CycleGAN, self).__init__()
        self.gen_A2B = generator_A2B
        self.gen_B2A = generator_B2A
        self.disc_A = discriminator_A
        self.disc_B = discriminator_B
        self.lambda_cycle = lambda_cycle

        # Define the loss functions
        self.criterion_gan = nn.BCELoss()
        self.criterion_cycle = nn.L1Loss()

    def forward(self, real_A, real_B):
        # Generate fake images
        fake_B = self.gen_A2B(real_A)
        fake_A = self.gen_B2A(real_B)

        # Cycle reconstruction
        cycle_A = self.gen_B2A(fake_B)
        cycle_B = self.gen_A2B(fake_A)

        # Discriminator outputs
        disc_real_A = self.disc_A(real_A)
        disc_fake_A = self.disc_A(fake_A.detach())
        disc_real_B = self.disc_B(real_B)
        disc_fake_B = self.disc_B(fake_B.detach())

        # Adversarial loss
        loss_gan_A = self.criterion_gan(disc_fake_A, torch.zeros_like(disc_fake_A))
        loss_gan_B = self.criterion_gan(disc_fake_B, torch.zeros_like(disc_fake_B))

        # Cycle consistency loss
        loss_cycle_A = self.criterion_cycle(cycle_A, real_A)
        loss_cycle_B = self.criterion_cycle(cycle_B, real_B)

        # Total loss
        loss_G = (
            loss_gan_A + loss_gan_B + self.lambda_cycle * (loss_cycle_A + loss_cycle_B)
        )
        loss_D_A = self.criterion_gan(
            disc_real_A, torch.ones_like(disc_real_A)
        ) 
        loss_D_B = self.criterion_gan(
            disc_real_B, torch.ones_like(disc_real_B)
        )

        return loss_G, loss_D_A, loss_D_B


# Training Loop
def train_cyclegan(dataloader_A, dataloader_B, num_epochs, men):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the generator and discriminator models
    gen_A2B = Generator(3, 3).to(device)
    gen_B2A = Generator(3, 3).to(device)
    disc_A = Discriminator(3).to(device)
    disc_B = Discriminator(3).to(device)

    # Initialize the CycleGAN model
    cyclegan = CycleGAN(gen_A2B, gen_B2A, disc_A, disc_B).to(device)

    # Define the optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(gen_A2B.parameters(), gen_B2A.parameters()),
        lr=2e-4,
        betas=(0.5, 0.999),
    )
    optimizer_D_A = torch.optim.Adam(disc_A.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(disc_B.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        for real_A, real_B in zip(dataloader_A, dataloader_B):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Update discriminators
            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()
            _, loss_D_A, loss_D_B = cyclegan(real_A, real_B)
            loss_D = loss_D_A + loss_D_B
            loss_D.backward(retain_graph=True)
            optimizer_D_A.step()
            optimizer_D_B.step()

            optimizer_G.zero_grad()
            loss_G, _, _ = cyclegan(real_A, real_B)
            loss_G.backward(retain_graph=True)
            optimizer_G.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss_G: {loss_G.item()}, Loss_D: {loss_D.item()}"
        )

    if men:
        torch.save(gen_A2B.state_dict(), "checkpoints/gen_A2B_men.pth")
        torch.save(gen_B2A.state_dict(), "checkpoints/gen_B2A_men.pth")
        torch.save(disc_A.state_dict(), "checkpoints/disc_A_men.pth")
        torch.save(disc_B.state_dict(), "checkpoints/disc_B_men.pth")
    else:
        torch.save(gen_A2B.state_dict(), "checkpoints/gen_A2B_women.pth")
        torch.save(gen_B2A.state_dict(), "checkpoints/gen_B2A_women.pth")
        torch.save(disc_A.state_dict(), "checkpoints/disc_A_women.pth")
        torch.save(disc_B.state_dict(), "checkpoints/disc_B_women.pth")

# Training

if __name__ == "__main__":
    train_cyclegan(men_no_glasses_loader, men_with_glasses_loader, num_epochs=100, men=True)
    print(f"Training of men completed!")
    train_cyclegan(men_with_glasses_loader, women_with_glasses_loader, num_epochs=100, men=False)
    print(f"Training of women completed!")
