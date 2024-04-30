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
from model import Encoder,Generator

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

    # Initialize the loss (binary cross entropy) function
    criterion = nn.BCELoss()

    # selecting the device
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

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
    
    # create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE,shuffle=True, num_workers=workers)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE,shuffle=True, num_workers=workers)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE,shuffle=True, num_workers=workers)

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

    # loading trained generator model
    netG = Generator(ngpu,NOISE_DIM,FEATURES_GEN,CHANNELS_IMG,).to(device)
    netG.load_state_dict(torch.load("./trained_models/generator.pth", map_location=device))
    netG.eval()

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

    # saving trained models 
    torch.save(encoder.state_dict(),"./trained_models/encoder.pth")

    ############################################################################################

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()