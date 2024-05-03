import random
import torch
import matplotlib.pyplot as plt
from model import Encoder, Generator
from PIL import Image
import numpy as np
import os
from split_dataset import (
    men_no_glasses,
    people_with_glasses,
    people_no_glasses,
    men_with_glasses,
    women_no_glasses,
    men_with_smile,
    people_with_hat,
    people_with_mus,
    people_no_mus,
)


# Function to generate an image from the given latent vector using the generator
def generate_image(latent_vector, generator):
    latent_vector = latent_vector.view(1, 100, 1, 1)  # Assuming latent_dim is 100
    generated_image = generator(latent_vector)
    return generated_image[0].detach().cpu().numpy().transpose(1, 2, 0)


def plot_combined_grid(image_files, output_filename):
    # Create a larger plot with 3 rows and however many columns
    num_files = len(image_files)
    rows = 3  # Each 3x3 grid has 3 rows
    cols = num_files  # Number of 3x3 grids to combine

    fig, axs = plt.subplots(rows, cols, figsize=(15, 15))  # Adjust size as needed
    axs = axs.ravel()  # Flatten for easy iteration

    for i, file_path in enumerate(image_files):
        img = Image.open(file_path)  # Load the 3x3 grid
        axs[i].imshow(np.asarray(img))
        axs[i].axis("off")  # Remove axis for clarity

    plt.tight_layout()
    plt.savefig(output_filename)  # Save the combined grid
    plt.close()  # Close the plot


# Function to plot a 3x3 grid of images
def plot_image_grid(images, grid_shape, filename):
    rows, cols = grid_shape
    fig, axs = plt.subplots(rows, cols, figsize=(6, 6))
    axs = axs.ravel()  # Flatten the array for easier indexing
    for i in range(len(images)):
        axs[i].imshow(images[i])
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    # Set manual seed for reproducibility
    manualSeed = 69
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device("cpu")  # Change to 'cuda' if using a GPU
    generator = Generator().to(device)
    encoder = Encoder().to(device)

    try:
        generator.load_state_dict(
            torch.load("./checkpoints/generator.pth", map_location=device)
        )
        encoder.load_state_dict(
            torch.load("./checkpoints/encoder.pth", map_location=device)
        )
        generator.eval()
        encoder.eval()
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    num_images = 9
    images = []

    # Create first grid with a combination of smile, hat, mus, and no mus
    # Reset images for each grid
    images.clear()

    # Perform vector arithmetic to generate a new latent vector
    for x in range(5):
        latent_vector = (
            encoder(men_with_smile[x])
            + encoder(people_with_hat[x])
            - encoder(people_with_hat[x])
            + encoder(people_with_mus[x])
            - encoder(people_no_mus[x])
        )

        latent_vector = latent_vector - 4 * 0.25 * torch.randn([1, 100], device=device)
        generated_image = generate_image(latent_vector, generator)
        images.append(generated_image)

        # Generate additional images for the 3x3 grid
        for i in range(8):
            latent_vector = latent_vector + 0.25 * torch.randn([1, 100], device=device)
            generated_image = generate_image(latent_vector, generator)
            images.append(generated_image)

        # Plot and save the 3x3 grid
        plot_image_grid(images, (3, 3), f"men_hat_{x}.png")

        # Create second grid with a combination of men with and without glasses, plus women no glasses
        images.clear()

    for x in range(5):
        latent_vector = (
            encoder(men_with_glasses[x])
            - encoder(men_no_glasses[x])
            + encoder(women_no_glasses[x])
        )

        latent_vector = latent_vector - 4 * 0.25 * torch.randn([1, 100], device=device)
        generated_image = generate_image(latent_vector, generator)
        images.append(generated_image)

        for i in range(8):
            latent_vector = latent_vector + 0.25 * torch.randn([1, 100], device=device)
            generated_image = generate_image(latent_vector, generator)
            images.append(generated_image)

        plot_image_grid(images, (3, 3), f"men_women_{x}.png")

        images.clear()

    for x in range(5):
        latent_vector = (
            encoder(men_no_glasses[x])
            + encoder(people_with_glasses[x])
            - encoder(people_no_glasses[x])
        )

        latent_vector = latent_vector - 4 * 0.25 * torch.randn([1, 100], device=device)
        generated_image = generate_image(latent_vector, generator)
        images.append(generated_image)

        for i in range(8):
            latent_vector = latent_vector + 0.25 * torch.randn([1, 100], device=device)
            generated_image = generate_image(latent_vector, generator)
            images.append(generated_image)

        plot_image_grid(images, (3, 3), f"men_people_{x}.png")
        images.clear()

    men_hat_files = [f"men_hat_{x}.png" for x in range(5)]
    men_people_files = [f"men_people_{x}.png" for x in range(5)]
    men_women_files = [f"men_women_{x}.png" for x in range(5)]

    plot_combined_grid(men_hat_files, "men_hat_grid.png")
    plot_combined_grid(men_people_files, "men_people_grid.png")
    plot_combined_grid(men_women_files, "men_women_grid.png")


if __name__ == "__main__":
    main()
