import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from model import Discriminator, Generator, Encoder, weights_init

def main():
    # set random seed for reproducibility, so that when everytime code is run, same results are produced
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.use_deterministic_algorithms(True)

    # defining hyperparameters (as given in DCGAN paper)
    LEARNING_RATE = 0.0002 
    BATCH_SIZE = 128 # mini-batch SGD
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3 # for RGB
    NOISE_DIM = 100
    NUM_EPOCHS = 20
    FEATURES_DISC = 64
    FEATURES_GEN = 64

    # number of workers for dataloader
    workers = 2
    # Beta1 hyperparameter for Adam optimizers
    beta1 = 0.5
    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    # for displaying the image
    def save_image(tensor, file_path):
        tensor = tensor.cpu().detach()

        # Normalize and convert to numpy
        tensor = tensor / 2 + 0.5  # Unnormalize
        npimg = tensor.numpy()

        # Check tensor dimensions
        if len(npimg.shape) == 3:  # If it’s a single image, transpose as (H, W, C)
            npimg = np.transpose(npimg, (1, 2, 0))
        elif len(npimg.shape) == 4:  # If it's a batch, create a grid
            npimg = torchvision.utils.make_grid(tensor, normalize=True).numpy()
            npimg = np.transpose(npimg, (1, 2, 0))

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save image
        plt.imsave(file_path, npimg)

    # pre-processing images, creates a pipeline for transforming images using dataloading step
    # taking dataset from local directory
    dataset_train = dset.ImageFolder(root="./dataset",transform=transforms.Compose([
                                transforms.Resize(IMAGE_SIZE),
                                transforms.CenterCrop(IMAGE_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    
    dataset_val = dset.ImageFolder(root="./dataset",transform=transforms.Compose([
                                transforms.Resize(IMAGE_SIZE),
                                transforms.CenterCrop(IMAGE_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    dataset_test = dset.ImageFolder(root="./dataset",transform=transforms.Compose([
                                transforms.Resize(IMAGE_SIZE),
                                transforms.CenterCrop(IMAGE_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))

    # making directory for saving results
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("trained_models", exist_ok=True)

    
    # create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE,shuffle=True, num_workers=workers)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE,shuffle=True, num_workers=workers)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE,shuffle=True, num_workers=workers)
    
    # selecting the device
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # create the generator
    netG = Generator(ngpu,NOISE_DIM,FEATURES_GEN,CHANNELS_IMG,).to(device)

    # handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # initialize weights
    netG.apply(weights_init)

    # Create the Discriminator
    netD = Discriminator(ngpu,FEATURES_DISC,CHANNELS_IMG).to(device)

    # handle multi-GPU if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # initialize all weights
    netD.apply(weights_init)

    # Initialize the loss (binary cross entropy) function
    criterion = nn.BCELoss()

    # create batch of latent vectors that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(64, NOISE_DIM, 1, 1, device=device)

    # putting real as 1 and fake as 0
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(beta1, 0.999))

    # Lists to keep track of progress
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(NUM_EPOCHS):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):

            # training discriminator involves maximizing the objective function log(D(x)) + log(1 - D(G(z)))
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, NOISE_DIM, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            # training generator involves minimizing log(1 - D(G(z))) which is equivalent to maximizing log(D(G(z))
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, NUM_EPOCHS, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))


            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                save_image(vutils.make_grid(fake, padding=2, normalize=True), f"results/fake_samples_epoch_{epoch}_iter_{iters}.png")

            iters += 1

    print('--------Finshed Training GAN--------')
    ############################################################################################

    # training the encoder for the given generator, for generating z vectors
    encoder = Encoder(CHANNELS_IMG, FEATURES_GEN, NOISE_DIM).to(device)

    # defining optimizer for encoder cnn
    opt_enc = optim.Adam(encoder.parameters())
    criterion_enc = nn.MSELoss()

    # defining best parameters for encoder
    best_loss = float('inf')
    best_lr = None
    best_beta1 = None

    # trying on different learning rates and beta1_values
    LEARNING_RATES = [0.0001, 0.0002, 0.0005]
    BETA1_VALUES = [0.5, 0.9]

    for lr in LEARNING_RATES:
        for beta1 in BETA1_VALUES:
            optimizer = optim.Adam(encoder.parameters(),lr=lr,betas=(beta1,0.999))

            for epoch in range(NUM_EPOCHS):
                encoder.train()
                total_loss = 0

                for batch_idx, (real, _) in enumerate(dataloader):
                    real = real.to(device)
                    optimizer.zero_grad()

                    # Generate fake images using fixed trained generator
                    noise = torch.randn(real.shape[0], NOISE_DIM, 1, 1).to(device)
                    fake = netG(noise)

                    # Encode fake images
                    encoded_fake = encoder(fake)

                    # Compute reconstruction loss, mse between input image feed to encoder and output image given by the gan generator
                    loss = criterion(encoded_fake, noise.view(noise.size(0), -1))
                    total_loss += loss.item()

                    # Backpropagation
                    loss.backward()
                    optimizer.step()

                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], LR: {lr}, Beta1: {beta1}, Avg Loss: {avg_loss}")

                # Validation
                encoder.eval()
                val_loss = 0

                with torch.no_grad():
                    for _, (real, _) in enumerate(dataloader_val):
                        real = real.to(device)
                        noise = torch.randn(real.shape[0], NOISE_DIM, 1, 1).to(device)
                        fake = netG(noise)
                        encoded_fake = encoder(fake)
                        val_loss += criterion(encoded_fake, noise.view(noise.size(0), -1)).item()

                avg_val_loss = val_loss / len(dataloader_val)
                print(f"Validation Loss: {avg_val_loss}")

                # Save best hyperparameters
                if avg_val_loss < best_loss:
                    best_loss = avg_val_loss
                    best_lr = lr
                    best_beta1 = beta1

    print(f"Best LR: {best_lr}, Best Beta1: {best_beta1}, Best Validation Loss: {best_loss}")
    print('--------Finshed Training Encoder--------')

    ############################################################################################

    # saving trained models
    torch.save(netG.state_dict(),"./trained_models/generator.pth")   
    torch.save(encoder.state_dict(),"./trained_models/encoder.pth")

    # plt.figure(figsize=(10,5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(G_losses,label="G")
    # plt.plot(D_losses,label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()