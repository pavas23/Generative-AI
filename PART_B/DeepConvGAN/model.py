import torch
import torch.nn as nn

# GAN Discriminator
class Discriminator(nn.Module):
    # discriminator takes a 64 * 64 image and downsamples the dimensions while increasing the number of channels to capture higher level features
    def __init__(self, channels_img, features_d):
        # as this class is inheriting from nn.Module class
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            # in_channels refers to number of channels in input feature map
            # out_channles refers to number of filters or feature maps produced after the conv operation
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)

# GAN Generator
class Generator(nn.Module):
    # generator takes a noise vector as input and transforms it into a fake img with desired number of channels and dimensions
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()

        # channels_noise is the number of channels in the input noise vector
        # channels_img is the number of channels in the output generated image
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            # weights are initialized from a zero-centered normal with standard deviation of 0.03
            nn.init.normal_(m.weight.data, 0.0, 0.02)

# CNN Encoder model
class Encoder(nn.Module):
    def __init__(self, channels_img, features_gen, noise_dim):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            # layer 1, input img: 64 * 64
            nn.Conv2d(channels_img, features_gen, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(features_gen),
            # layer 2, input img: 32 * 32
            nn.Conv2d(features_gen, features_gen*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(features_gen*2),
            # layer 3, input img: 16 * 16
            nn.Conv2d(features_gen*2, features_gen*4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(features_gen*4),
            # layer 4, input img: 8 * 8
            nn.Conv2d(features_gen*4, features_gen*8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(features_gen*8),
            # layer 5, input img: 4 * 4
            nn.Conv2d(features_gen*8, features_gen*16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(features_gen*16),
            # layer 6, input img: 2 * 2
            nn.Conv2d(features_gen*16, features_gen*32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(features_gen*32),
            # layer 7, noise of dim 1000
            nn.Conv2d(features_gen*32, noise_dim, kernel_size=4, stride=2, padding=0),
            nn.Flatten()
        )

    def forward(self, x):
        return self.encoder(x)
