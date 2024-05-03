import random
import torch
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
    people_with_mus,
    people_no_mus,
)


# Function to generate an image from the given latent vector using the generator
def generate_image(latent_vector, generator):
    latent_vector = latent_vector.view(1, 100, 1, 1)  # Assuming latent_dim is 100
    generated_image = generator(latent_vector)
    return generated_image[0].detach().cpu().numpy().transpose(1, 2, 0)


# Function to plot a grid of images
def plot_image_grid(images, grid_shape, filename):
    rows, cols = grid_shape
    _, axs = plt.subplots(rows, cols, figsize=(6, 6))
    axs = axs.ravel()  # Flatten the array for easier indexing
    for i in range(len(images)):
        axs[i].imshow(images[i])
        axs[i].axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Main logic to generate a 3x3 grid of images
def main():
    # Set manual seed for reproducibility
    manualSeed = 69
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # Load generator and encoder
    generator = Generator()
    generator.load_state_dict(torch.load("./checkpoints/generator.pth"))
    generator.eval()

    encoder = Encoder()
    encoder.load_state_dict(torch.load("./checkpoints/encoder.pth"))
    encoder.eval()

    # Create a 3x3 grid of generated images
    num_images = 9
    dataset_size = 5  # Number of available images
    images = []

    for i in range(num_images):
        # Use modulo arithmetic to avoid index out of range errors
        x = i % dataset_size
        y = (i + 1) % dataset_size  # To create some variety in vector arithmetic

        # Perform vector arithmetic to generate a new latent vector
        latent_vector = (
            encoder(men_with_smile[x])
            + encoder(people_with_hat[x])
            - encoder(people_with_hat[y])
            + encoder(people_with_mus[x])
            - encoder(people_no_mus[x])
        )

        # Generate an image using the new latent vector
        generated_image = generate_image(latent_vector, generator)
        images.append(generated_image)

    for i in range(num_images):
        # Use modulo arithmetic to avoid index out of range errors
        x = i % dataset_size
        y = (i + 1) % dataset_size  # To create some variety in vector arithmetic

        # Perform vector arithmetic to generate a new latent vector
        latent_vector = (
            encoder(men_with_glasses[x])
            - encoder(men_no_glasses[y])
            + encoder(women_no_glasses[x])
        )

        # Generate an image using the new latent vector
        generated_image = generate_image(latent_vector, generator)
        images.append(generated_image)

    for i in range(num_images):
        # Use modulo arithmetic to avoid index out of range errors
        x = i % dataset_size
        y = (i + 1) % dataset_size  # To create some variety in vector arithmetic

        # Perform vector arithmetic to generate a new latent vector
        latent_vector = (
            encoder(men_no_glasses[x])
            + encoder(people_with_glasses[x])
            - encoder(people_no_glasses[x])
        )

        # Generate an image using the new latent vector
        generated_image = generate_image(latent_vector, generator)
        images.append(generated_image)

    # Plot and save the 3x3 grid
    plot_image_grid(images, (3, 3), "3x3_image_grid.png")


if __name__ == "__main__":
    main()
