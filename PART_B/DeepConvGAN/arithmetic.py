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
from split_dataset import (
    men_no_glasses,
    people_with_glasses,
    people_no_glasses,
    men_with_glasses,
    women_no_glasses,
    men_with_smile,
    people_with_hat,
    people_no_hat,
    people_with_mus,
    people_no_mus,
)


def main():
    manualSeed = 69
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda:1")
    netG = Generator().to(device)
    netG.load_state_dict(torch.load("./checkpoints/generator.pth"))
    netG.eval()
    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("./checkpoints/encoder.pth"))
    encoder.eval()

    insides = []
    outsides = []
    for i in range(5):
        print(men_no_glasses[i].shape)
        print(encoder(men_no_glasses[i]).shape)
        inside = (
            encoder(men_no_glasses[i])
            + encoder(people_with_glasses[i])
            - encoder(people_no_glasses[i])
        )
        outside = encoder(
            men_no_glasses[i] + people_with_glasses[i] - people_no_glasses[i]
        )
        inside_image = netG(inside)
        outside_image = netG(outside)
        print(inside_image.shape)
        print(outside_image.shape)
        insides.append(inside_image)
        outsides.append(outside_image)

    fig, axs = plt.subplots(5, 5, figsize=(9, 15))
    for i in range(5):
        axs[i, 0].imshow(men_no_glasses[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 0].axis("off")
        axs[i, 1].imshow(
            people_with_glasses[i].detach().cpu().numpy().transpose(1, 2, 0)
        )
        axs[i, 1].axis("off")
        axs[i, 2].imshow(people_no_glasses[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 2].axis("off")
        axs[i, 3].imshow(insides[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 3].axis("off")
        axs[i, 4].imshow(outsides[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 4].axis("off")
    plt.tight_layout()
    plt.savefig(f"results_fin/men_with_glass.png")
    plt.close()

    insides = []
    outsides = []
    for i in range(5):
        print(men_no_glasses[i].shape)
        print(encoder(men_no_glasses[i]).shape)
        inside = (
            encoder(men_with_glasses[i])
            - encoder(men_no_glasses[i])
            + encoder(women_no_glasses[i])
        )
        outside = encoder(men_with_glasses[i] - men_no_glasses[i] + women_no_glasses[i])
        inside_image = netG(inside)
        outside_image = netG(outside)
        print(inside_image.shape)
        print(outside_image.shape)
        insides.append(inside_image)
        outsides.append(outside_image)

    fig, axs = plt.subplots(5, 5, figsize=(9, 15))
    for i in range(5):
        axs[i, 0].imshow(men_with_glasses[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 0].axis("off")
        axs[i, 1].imshow(men_no_glasses[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 1].axis("off")
        axs[i, 2].imshow(women_no_glasses[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 2].axis("off")
        axs[i, 3].imshow(insides[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 3].axis("off")
        axs[i, 4].imshow(outsides[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 4].axis("off")
    plt.tight_layout()
    plt.savefig(f"results_fin/women_with_glass.png")
    plt.close()

    insides = []
    outsides = []
    for i in range(5):
        print(men_with_smile[i].shape)
        print(encoder(men_with_smile[i]).shape)
        inside = (
            encoder(men_with_smile[i])
            + encoder(people_with_hat[i])
            - encoder(people_no_hat[i])
            + encoder(people_with_mus[i])
            - encoder(people_no_mus[i])
        )
        outside = encoder(
            men_with_smile[i]
            + people_with_hat[i]
            - people_no_hat[i]
            + people_with_mus[i]
            - people_no_mus[i]
        )
        inside_image = netG(inside)
        outside_image = netG(outside)
        print(inside_image.shape)
        print(outside_image.shape)
        insides.append(inside_image)
        outsides.append(outside_image)

    fig, axs = plt.subplots(5, 7, figsize=(9, 21))
    for i in range(5):
        axs[i, 0].imshow(men_with_smile[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 0].axis("off")
        axs[i, 1].imshow(people_with_hat[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 1].axis("off")
        axs[i, 2].imshow(people_no_hat[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 2].axis("off")
        axs[i, 3].imshow(people_with_mus[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 3].axis("off")
        axs[i, 4].imshow(people_no_mus[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 4].axis("off")
        axs[i, 5].imshow(insides[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 5].axis("off")
        axs[i, 6].imshow(outsides[i].detach().cpu().numpy().transpose(1, 2, 0))
        axs[i, 6].axis("off")
    plt.tight_layout()
    plt.savefig(f"results_fin/men_with_hat_smile_mustache.png")
    plt.close()


if __name__ == "__main__":
    main()
