import torch
import os
import random
from PIL import Image
import csv
from model import Encoder, Generator
import numpy as np

# TODO: change this to perform on test dataset only !!!!!

# device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# defining hyperparameters (as given in DCGAN paper)
LEARNING_RATE = 0.0002 
BATCH_SIZE = 128 # mini-batch SGD
IMAGE_SIZE = 64
CHANNELS_IMG = 3 # for RGB
NOISE_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
ngpu = 1

# loading trained encoder and generator models
encoder = Encoder(CHANNELS_IMG, FEATURES_GEN, NOISE_DIM).to(device)
encoder.load_state_dict(torch.load("./trained_models/encoder.pth", map_location=device))
encoder.eval()

gen = Generator(ngpu,NOISE_DIM,FEATURES_GEN, CHANNELS_IMG).to(device)
gen.load_state_dict(torch.load("./trained_models/generator.pth", map_location=device))
gen.eval()

list_attr_path = os.path.join("./dataset","list_attr_celeba.txt")
with open(list_attr_path,"r") as file:
    lines = file.readlines()

header = lines[1].split()

# Extract image filenames and corresponding attributes
image_filenames = []
attributes = {}
for line in lines[2:]:  # Skip the first two lines (header and number of images)
    parts = line.split()
    image_filename = parts[0]
    image_filenames.append(image_filename)
    attr_dict = {header[i]: parts[i + 1] for i in range(len(header))}  # Create a dictionary for attributes
    attributes[image_filename] = attr_dict

# define the desired attributes for sampling
men_without_glasses_attributes = {
    "Male": 1,  
    "Eyeglasses": -1 
}
people_with_glasses_attributes = {
    "Eyeglasses":1
}
people_without_glasses_attributes = {
    "Eyeglasses":-1
}

# Function to sample images
def sample_images_for_a_category(desired_attrs, sampled_images):
    # Get the subset of image filenames that satisfy all desired attributes
    subset_indices = None
    for image_attr, attr_value in desired_attrs.items():
        if subset_indices is None:
            subset_indices = get_subset(attributes, image_attr, attr_value == "1")
        else:
            indices = get_subset(attributes, image_attr, attr_value == "1")
            subset_indices = np.intersect1d(subset_indices, indices)

    # Randomly sample images from the subset
    num_samples_per_category = 3
    if len(subset_indices) >= num_samples_per_category:
        sampled_indices = random.sample(subset_indices.tolist(), num_samples_per_category)
    else:
        sampled_indices = subset_indices

    # Load and display the sampled images
    for index in sampled_indices:
        img = image_filenames[index]
        image_path = os.path.join("./dataset/celeba", "img_align_celeba", img)
        image = Image.open(image_path)
        image.show()



# Sample images for men without glasses
sampled_images_men_without_glasses = {attr: [] for attr in men_without_glasses_attributes}
sample_images_for_a_category(men_without_glasses_attributes, sampled_images_men_without_glasses)

print(len(sampled_images_men_without_glasses))

# Sample images for people with glasses
sampled_images_people_with_glasses = {attr: [] for attr in people_with_glasses_attributes}
# sample_images_for_a_category(people_with_glasses_attributes, sampled_images_people_with_glasses)

# Sample images for people without glasses
sampled_images_people_without_glasses = {attr: [] for attr in people_without_glasses_attributes}
sample_images_for_a_category(people_without_glasses_attributes, sampled_images_people_without_glasses)

# function to perform vector arithmetic in z space
def perform_vector_arithmetic_in_z_space(z_vectors):
    # Vector arithmetic operations
    # Vector arithmetic 1: Men without glasses + People with glasses - People without glasses
    # Vector arithmetic 2: Men with glasses - Men without glasses + Women without glasses
    # Vector arithmetic 3: Smiling Men + People with Hat - People with Hat + People with Mustache - People without Mustache
    translated_z_vectors = z_vectors[0] + z_vectors[1] - z_vectors[2]
    return translated_z_vectors

# Sample three images from each category and obtain their z vectors using the encoder
def sample_z_vectors(images):
    z_vectors = []
    for image in images:
        image = image.to(device).unsqueeze(0)
        z_vector = encoder(image)
        z_vectors.append(z_vector)
    return z_vectors

# Perform vector arithmetic and generate translated images
def generate_translated_images(z_vectors):
    translated_images = []
    for i in range(5):  # Generate 5 outputs
        # Perform vector arithmetic in z space
        translated_z_vectors = perform_vector_arithmetic_in_z_space(z_vectors)
        # Generate images from translated z vectors
        translated_image = gen(translated_z_vectors)
        translated_images.append(translated_image)
    return translated_images

# Sample three images from each category (Men without glasses, People with glasses, People without glasses) and obtain their z vectors
# Replace the placeholders with actual images
men_without_glasses_images = ...
people_with_glasses_images = ...
people_without_glasses_images = ...

men_without_glasses_z_vectors = sample_z_vectors(men_without_glasses_images)
people_with_glasses_z_vectors = sample_z_vectors(people_with_glasses_images)
people_without_glasses_z_vectors = sample_z_vectors(people_without_glasses_images)

# Perform vector arithmetic and generate translated images
translated_images_1 = generate_translated_images([men_without_glasses_z_vectors, people_with_glasses_z_vectors, people_without_glasses_z_vectors])

# Repeat the same process for other vector arithmetic operations

# Visualize the translated images
for i, translated_image in enumerate(translated_images_1):
    print(f"Translated Image {i+1}")
    #Visualize the translated image
