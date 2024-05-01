import random
import torch
import torch.nn.parallel
import torch.utils.data
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

    netG = Generator()
    netG.load_state_dict(torch.load("./checkpoints/generator.pth"))
    netG.eval()
    encoder = Encoder()
    encoder.load_state_dict(torch.load("./checkpoints/encoder.pth"))
    encoder.eval()

    insides = []
    outsides = []
    for i in range(5):
        if i == 4:
            x, y = 4, 0
        else:
            x, y = i, i + 1
        inside = (
            encoder(men_with_smile[x])
            + encoder(people_with_hat[x])
            - encoder(people_with_hat[y])
            + encoder(people_with_mus[x])
            - encoder(people_no_mus[x])
        )
        outside = encoder(
            men_with_smile[x]
            + people_with_hat[x]
            - people_with_hat[y]
            + people_with_mus[x]
            - people_no_mus[x]
        )
        inside_image = netG(inside.view(-1, 100, 1, 1))
        outside_image = netG(outside.view(-1, 100, 1, 1))
        insides.append(inside_image)
        outsides.append(outside_image)
        print(f"{i+1}/5 Images")

    _, axs = plt.subplots(5, 7, figsize=(9, 21))
    for i in range(5):
        if i == 4:
            x, y = 4, 0
        else:
            x, y = i, i+1

        axs[i, 0].imshow(
            men_with_smile[x].view(3, 64, 64).detach().cpu().numpy().transpose(1, 2, 0)
        )
        axs[i, 0].axis("off")
        axs[i, 1].imshow(
            people_with_hat[x].view(3, 64, 64).detach().cpu().numpy().transpose(1, 2, 0)
        )
        axs[i, 1].axis("off")
        axs[i, 2].imshow(
            people_with_hat[y].view(3, 64, 64).detach().cpu().numpy().transpose(1, 2, 0)
        )
        axs[i, 2].axis("off")
        axs[i, 3].imshow(
            people_with_mus[x].view(3, 64, 64).detach().cpu().numpy().transpose(1, 2, 0)
        )
        axs[i, 3].axis("off")
        axs[i, 4].imshow(
            people_no_mus[x].view(3, 64, 64).detach().cpu().numpy().transpose(1, 2, 0)
        )
        axs[i, 4].axis("off")
        axs[i, 5].imshow(
            insides[i].view(3, 64, 64).detach().cpu().numpy().transpose(1, 2, 0)
        )
        axs[i, 5].axis("off")
        axs[i, 6].imshow(
            outsides[i].view(3, 64, 64).detach().cpu().numpy().transpose(1, 2, 0)
        )
        axs[i, 6].axis("off")
    plt.tight_layout()
    plt.savefig(f"men_with_hat_smile_mustache.png")
    plt.close()
    print(f"Image saved")


if __name__ == "__main__":
    main()
