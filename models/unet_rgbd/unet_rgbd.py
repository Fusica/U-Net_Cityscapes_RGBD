import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet_RGBD(nn.Module):
    def __init__(self, num_features=64, use_bn=True, **kwargs):
        super(UNet_RGBD, self).__init__()
        self.use_bn = use_bn
        self.num_features = num_features

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

        self.dconv_up3 = DoubleConv(512 + 256, 256)
        self.dconv_up2 = DoubleConv(256 + 128, 128)
        self.dconv_up1 = DoubleConv(128 + 64, self.num_features)

        # attention
        self.attention_rgb = self.attention(512)
        self.attention_d = self.attention(512)


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
        if depth is not None:
            conv1_d = self.dconv_down1_d(depth.unsqueeze(1))
            x_d = self.maxpool(conv1_d)
            conv2_d = self.dconv_down2_d(x_d)
            x_d = self.maxpool(conv2_d)
            conv3_d = self.dconv_down3_d(x_d)
            x_d = self.maxpool(conv3_d)
            x_d = self.dconv_down4_d(x_d)

            # Attention and fusion
            x_rgb_attention = self.attention_rgb(x_rgb)
            x_d_attention = self.attention_d(x_d)
            x = torch.mul(x_rgb, x_rgb_attention) + torch.mul(x_d, x_d_attention)
        else:
            x = x_rgb

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

        return x

    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()

        return nn.Sequential(pool_attention, conv_attention, activate)

    # TODO can modify here
    def random_init_params(self):
        return self.parameters()
