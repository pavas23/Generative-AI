import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels, base_channels, time_emb_dim=10):
        super(UNet, self).__init__()
        # Time embedding to integrate timestep information
        self.time_emb = nn.Linear(1, time_emb_dim)

        # Define your downsampling layers
        self.down1 = self.conv_block(in_channels + time_emb_dim, base_channels)
        self.down2 = self.conv_block(base_channels, base_channels * 2)
        self.down3 = self.conv_block(base_channels * 2, base_channels * 4)

        self.middle = self.conv_block(base_channels * 4, base_channels * 8)

        self.up3 = self.up_block(base_channels * 8, base_channels * 4)
        self.up2 = self.up_block(base_channels * 4, base_channels * 2)
        self.up1 = self.up_block(base_channels * 2, base_channels)

        self.final_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels),
        )

    def forward(self, x, t=None):
        if t is not None:
            # Embed the timestep and add it to the input
            t_emb = self.time_emb(t).unsqueeze(-1).unsqueeze(-1)
            x = torch.cat([x, t_emb.repeat(1, 1, x.size(2), x.size(3))], dim=1)

        d1 = self.down1(x)
        d2 = self.down2(F.max_pool2d(d1, 2))
        d3 = self.down3(F.max_pool2d(d2, 2))

        middle = self.middle(F.max_pool2d(d3, 2))

        u3 = self.up3(middle)
        u2 = self.up2(torch.cat([u3, d3], dim=1))
        u1 = self.up1(torch.cat([u2, d2], dim=1))

        out = self.final_conv(torch.cat([u1, d1], dim=1))
        return out