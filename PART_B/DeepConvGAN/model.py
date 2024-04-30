import torch
import torch.nn as nn

# GAN Discriminator
class Discriminator(nn.Module):
    def __init__(self,ngpu,FEATURES_DISC,CHANNELS_IMG):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is ``(CHANNELS_IMG) x 64 x 64``
            nn.Conv2d(CHANNELS_IMG, FEATURES_DISC, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(FEATURES_DISC) x 32 x 32``
            nn.Conv2d(FEATURES_DISC, FEATURES_DISC * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_DISC * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(FEATURES_DISC*2) x 16 x 16``
            nn.Conv2d(FEATURES_DISC * 2, FEATURES_DISC * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_DISC * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(FEATURES_DISC*4) x 8 x 8``
            nn.Conv2d(FEATURES_DISC * 4, FEATURES_DISC * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_DISC * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(FEATURES_DISC*8) x 4 x 4``
            nn.Conv2d(FEATURES_DISC * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


# GAN Generator
class Generator(nn.Module):
    def __init__(self, ngpu, NOISE_DIM,FEATURES_GEN,CHANNELS_IMG):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( NOISE_DIM, FEATURES_GEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(FEATURES_GEN * 8),
            nn.ReLU(True),
            # state size. ``(FEATURES_GEN*8) x 4 x 4``
            nn.ConvTranspose2d(FEATURES_GEN * 8, FEATURES_GEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN * 4),
            nn.ReLU(True),
            # state size. ``(FEATURES_GEN*4) x 8 x 8``
            nn.ConvTranspose2d( FEATURES_GEN * 4, FEATURES_GEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN * 2),
            nn.ReLU(True),
            # state size. ``(FEATURES_GEN*2) x 16 x 16``
            nn.ConvTranspose2d( FEATURES_GEN * 2, FEATURES_GEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURES_GEN),
            nn.ReLU(True),
            # state size. ``(FEATURES_GEN) x 32 x 32``
            nn.ConvTranspose2d( FEATURES_GEN, CHANNELS_IMG, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(CHANNELS_IMG) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)

# custom weights initialization called for generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

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
            nn.Conv2d(features_gen*32, noise_dim, kernel_size=2, stride=1, padding=0),  # Adjusted kernel size
            nn.Flatten()
        )

    def forward(self, x):
        return self.encoder(x)
