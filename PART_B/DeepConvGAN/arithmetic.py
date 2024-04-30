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

def initialize_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def initialize_models(device):
    netG = Generator().to(device)
    netG.load_state_dict(torch.load("./checkpoints/generator.pth"))
    netG.eval()

    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load("./checkpoints/encoder.pth"))
    encoder.eval()

    return netG, encoder

def plot_images(input_sets, file_name, subplot_shape, figsize=(9, 15)):
    fig, axs = plt.subplots(subplot_shape[0], subplot_shape[1], figsize=figsize)
    for i, input_set in enumerate(input_sets):
        for j, image_tensor in enumerate(input_set):
            image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
            axs[i, j].imshow(image)
            axs[i, j].axis("off")
    plt.tight_layout()
    plt.savefig(f"results_fin/{file_name}.png")
    plt.close()


def generate_and_plot_images(
    device, netG, encoder, input_data, operations, file_name, subplot_shape
):
    insides = []
    outsides = []
    input_sets = []

    for i in range(5):
        inside_code = operations["inside"](i)
        outside_code = operations["outside"](i)

        inside_image = netG(inside_code)
        outside_image = netG(outside_code)

        insides.append(inside_image)
        outsides.append(outside_image)

        input_sets.append(input_data["inputs"][i] + [insides[i], outsides[i]])

    plot_images(input_sets, file_name, subplot_shape)


def main():
    initialize_seed(69)

    device = torch.device("cuda:1")
    netG, encoder = initialize_models(device)

    operations_men_with_glasses = {
        "inside": lambda i: (
            encoder(men_no_glasses[i])
            + encoder(people_with_glasses[i])
            - encoder(people_no_glasses[i])
        ),
        "outside": lambda i: (
            men_no_glasses[i] + people_with_glasses[i] - people_no_glasses[i]
        ),
    }

    input_data_men_with_glasses = {
        "inputs": [
            [men_no_glasses[i], people_with_glasses[i], people_no_glasses[i]]
            for i in range(5)
        ]
    }
    generate_and_plot_images(
        device,
        netG,
        encoder,
        input_data_men_with_glasses,
        operations_men_with_glasses,
        "men_with_glass",
        (5, 5),
    )

    operations_women_with_glasses = {
        "inside": lambda i: (
            encoder(men_with_glasses[i])
            - encoder(men_no_glasses[i])
            + encoder(women_no_glasses[i])
        ),
        "outside": lambda i: (
            men_with_glasses[i] - men_no_glasses[i] + women_no_glasses[i]
        ),
    }

    input_data_women_with_glasses = {
        "inputs": [
            [men_with_glasses[i], men_no_glasses[i], women_no_glasses[i]]
            for i in range(5)
        ]
    }
    generate_and_plot_images(
        device,
        netG,
        encoder,
        input_data_women_with_glasses,
        operations_women_with_glasses,
        "women_with_glass",
        (5, 5),
    )
    
    operations_men_with_hat_smile_mustache = {
        "inside": lambda i: (
            encoder(men_with_smile[i])
            + encoder(people_with_hat[i])
            - encoder(people_no_hat[i])
            + encoder(people_with_mus[i])
            - encoder(people_no_mus[i])
        ),
        "outside": lambda i: (
            men_with_smile[i]
            + people_with_hat[i]
            - people_no_hat[i]
            + people_with_mus[i]
            - people_no_mus[i]
        ),
    }

    input_data_men_with_hat_smile_mustache = {
        "inputs": [
            [
                men_with_smile[i],
                people_with_hat[i],
                people_no_hat[i],
                people_with_mus[i],
                people_no_mus[i],
            ]
            for i in range(5)
        ]
    }
    generate_and_plot_images(
        device,
        netG,
        encoder,
        input_data_men_with_hat_smile_mustache,
        operations_men_with_hat_smile_mustache,
        "men_with_hat_smile_mustache",
        (5, 7),
    )

if __name__ == "__main__":
    main()
