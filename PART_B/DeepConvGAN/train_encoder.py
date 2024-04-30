import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from model import Encoder, Generator


def main():
    manualSeed = 69
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

    dataset_train = dset.ImageFolder(
        root="../T/dataset",
        transform=transforms.Compose(
            [
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ]
        ),
    )

    dataset_val = dset.ImageFolder(
        root="../T/dataset",
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
    netG.load_state_dict(torch.load("./checkpoints/generator.pth"))
    netG.eval()

    criterion = nn.MSELoss()
    encoder = Encoder().to(device)
    optimizer = optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))
    print("Starting Training Loop...")
    for epoch in range(10):
        # encoder.train()
        # total_loss = 0

        # for batch_idx, (real, _) in enumerate(dataloader):
        #     real = real.to(device)
        #     noise = encoder(real).view(-1, 100, 1, 1)
        #     fake = netG(noise)
        #     loss = criterion(real, fake)
        #     total_loss += loss.item()

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     avg_loss = total_loss / len(dataloader)
        #     print(f"Epoch [{epoch+1}/{20}] Loss: {avg_loss}")

        encoder.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (real, _) in enumerate(dataloader_val):
                real = real.to(device)
                noise = encoder(real).view(-1, 100, 1, 1)
                fake = netG(noise)
                val_loss += criterion(real, fake).item()
        avg_val_loss = val_loss / len(dataloader_val)
        print(f"Validation Loss: {avg_val_loss}")

        fig, axs = plt.subplots(2, 5, figsize=(15, 6))
        for j in range(5):
            axs[0, j].imshow(real[j].detach().cpu().numpy().transpose(1, 2, 0))
            axs[0, j].axis("off")
            axs[1, j].imshow(fake[j].detach().cpu().numpy().transpose(1, 2, 0))
            axs[1, j].axis("off")

        plt.tight_layout()
        plt.savefig(f"results_enc/images_{epoch}.png")
        plt.close()
        print(f"Images saved for epoch {epoch}")

    torch.save(encoder.state_dict(), "./checkpoints/encoder.pth")


if __name__ == "__main__":
    main()
