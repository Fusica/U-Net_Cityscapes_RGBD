import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes, use_bn=True):
        super(UNet, self).__init__()
        self.use_bn = use_bn

        # RGB branch
        self.dconv_down1_rgb = DoubleConv(3, 64)
        self.dconv_down2_rgb = DoubleConv(64, 128)
        self.dconv_down3_rgb = DoubleConv(128, 256)
        self.dconv_down4_rgb = DoubleConv(256, 512)

        # Depth branch
        self.dconv_down1_d = DoubleConv(1, 64)
        self.dconv_down2_d = DoubleConv(64, 128)
        self.dconv_down3_d = DoubleConv(128, 256)
        self.dconv_down4_d = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = DoubleConv(512 + 512, 256)
        self.dconv_up2 = DoubleConv(256 + 256, 128)
        self.dconv_up1 = DoubleConv(128 + 128, 64)

        self.conv_last = nn.Conv2d(64, num_classes, 1)

    def forward(self, rgb, depth=None):
        # Down RGB
        conv1_rgb = self.dconv_down1_rgb(rgb)
        x_rgb = self.maxpool(conv1_rgb)
        conv2_rgb = self.dconv_down2_rgb(x_rgb)
        x_rgb = self.maxpool(conv2_rgb)
        conv3_rgb = self.dconv_down3_rgb(x_rgb)
        x_rgb = self.maxpool(conv3_rgb)
        x_rgb = self.dconv_down4_rgb(x_rgb)

        # Down Depth
        conv1_d = self.dconv_down1_d(depth)
        x_d = self.maxpool(conv1_d)
        conv2_d = self.dconv_down2_d(x_d)
        x_d = self.maxpool(conv2_d)
        conv3_d = self.dconv_down3_d(x_d)
        x_d = self.maxpool(conv3_d)
        x_d = self.dconv_down4_d(x_d)

        # Attention and fusion
        x_rgb_attention = self.attention(x_rgb)
        x_d_attention = self.attention(x_d)
        x = torch.mul(x_rgb, x_rgb_attention) + torch.mul(x_d, x_d_attention)

        # Up
        x = self.upsample(x)
        x = torch.cat([x, conv3_rgb], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2_rgb], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1_rgb], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out, out  # Return the final output and additional features for compatibility

    def attention(self, x):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(x.size(1), x.size(1), kernel_size=1)
        activate = nn.Sigmoid()
        return nn.Sequential(pool_attention, conv_attention, activate)

    @property
    def num_features(self):
        return self.out_channels

    def random_init_params(self):
        return self.parameters()

    def fine_tune_params(self):
        return self.parameters()
