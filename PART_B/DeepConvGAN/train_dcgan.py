import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import Discriminator, Generator

def main():
    manualSeed = 69
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

    dataset_train = dset.ImageFolder(
        root="./dataset",
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        ),
    )

    dataset_val = dset.ImageFolder(
        root="./dataset",
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        ),
    )

    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=64, shuffle=True
    )

    device = torch.device("cuda:1")
    netG = Generator().to(device)
    netD = Discriminator().to(device)
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(5, 100, 1, 1, device=device)
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999)) # stable training at beta = 0.5
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    iters = 0
    print("Starting Training Loop...")
    for epoch in range(10):
        for i, data in enumerate(dataloader, 0):
            bs = data[0].shape[0]
            noise = torch.randn(bs, 100, 1, 1, device=device)
            real = data[0].to(device)
            fake = netG(noise)
            loss_D = 0.5 * (
                criterion(
                    netD(real), torch.ones(bs, 1, device=device).view(-1, 1, 1, 1)
                )
                + criterion(
                    netD(fake), torch.zeros(bs, 1, device=device).view(-1, 1, 1, 1)
                )
            )
            netD.zero_grad()
            loss_D.backward()
            optimizerD.step()

            noise = torch.randn(bs, 100, 1, 1, device=device)
            fake = netG(noise)
            loss_G = criterion(
                netD(fake), torch.ones(bs, 1, device=device).view(-1, 1, 1, 1)
            )
            netG.zero_grad()
            loss_G.backward()
            optimizerG.step()
            print(
                f"Epoch: {epoch}, Iteration: {i}, Loss D: {loss_D.item()}, Loss G: {loss_G.item()}"
            )

            if i == len(dataloader) - 1:
                with torch.no_grad():
                    image = netG(fixed_noise).detach().cpu()
                _, axs = plt.subplots(1, 5, figsize=(15, 3))
                for j in range(5):
                    axs[j].imshow(image[j].detach().cpu().numpy().transpose(1, 2, 0))
                    axs[j].axis("off")
                plt.tight_layout()
                plt.savefig(f"results_gan/images_{epoch}_{i}.png")
                plt.close()
                print(f"Images saved for epoch {epoch}")
            iters += 1
    torch.save(netG.state_dict(), "./checkpoints/generator.pth")


if __name__ == "__main__":
    main()