import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        self.down = down
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if self.down:
            x = self.pool(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down1 = UNetBlock(in_channels, 64, downsample=True)
        self.down2 = UNetBlock(64, 128, downsample=True)
        self.down3 = UNetBlock(128, 256, downsample=True)
        self.down4 = UNetBlock(256, 512, downsample=True)
        self.bottleneck = UNetBlock(512, 1024)
        self.up1 = UNetBlock(1024, 512)
        self.up2 = UNetBlock(512, 256)
        self.up3 = UNetBlock(256, 128)
        self.up4 = UNetBlock(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.bottleneck(x4)
        x = self.up1(torch.cat([x4, x], dim=1))
        x = self.up2(torch.cat([x3, x], dim=1))
        x = self.up3(torch.cat([x2, x], dim=1))
        x = self.up4(torch.cat([x1, x], dim=1))
        return self.out(x)
