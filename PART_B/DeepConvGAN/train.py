import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights

# selecting the device
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

# pre-processing images, creates a pipeline for transforming images using dataloading step
transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(), # converts image to pytorch tensor
        # sets mean and std for each channel to 0.5
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]
        ),
    ]
)

# taking dataset from loacal directory
dataset_train = datasets.ImageFolder(root="./dataset/celeba_train", transform=transforms)
dataset_val = datasets.ImageFolder(root="./dataset/celeba_val", transform=transforms)
dataset_test = datasets.ImageFolder(root="./dataset/celeba_test", transform=transforms)

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

# initialize weights for both generator and discriminator
initialize_weights(gen)
initialize_weights(disc)

# using Adam optimizer as per DCGAN paper
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

# setting the loss function as binary cross entropy loss
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)

# for logging real and fake images during training using tensorboard
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    # for each epoch, it iterates on every batch one by one
    for batch_idx, (real, _) in enumerate(dataloader_train):
        real = real.to(device)
        noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1).to(device)
        fake = gen(noise)

        # training discriminator involves maximizing the objective function log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2 # taking average loss
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        ## training generator involves minimizing log(1 - D(G(z))) which is equivalent to maximizing log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
                f"Epoch: [{epoch}/{NUM_EPOCHS}] Batch: [{batch_idx}/{len(dataloader_train)}] Loss D: [{loss_disc:.4f}], loss G: [{loss_gen:.4f}]"
            )

            with torch.no_grad():
                # at every step for fixed noise, check how to images produced by generator improves over time
                fake = gen(fixed_noise)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            step += 1