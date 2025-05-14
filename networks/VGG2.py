#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch

def initialize_weights_to_zero(m):
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class VGG3(nn.Module):
    def __init__(self, args, level, estimate_depth=False, attention='none'):
        super(VGG3, self).__init__()
        # print('estimate_depth: ', estimate_depth)
        self.args = args
        self.level = level
        self.estimate_depth = estimate_depth
        self.attention = attention
        # self.enc_feature = []

        vgg16 = torchvision.models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('/ws/external/checkpoints/pretrained/vgg16-397923af.pth'))

        # load CNN from VGG16, the first three block
        # Feature Encoder
        self.conv0 = vgg16.features[0]
        self.conv2 = vgg16.features[2]  # \\64 [H/2, W/2]
        self.conv5 = vgg16.features[5]  #
        self.conv7 = vgg16.features[7]  # \\128 [H/4, W/4]
        self.conv10 = vgg16.features[10]
        self.conv12 = vgg16.features[12]
        self.conv14 = vgg16.features[14]  # \\ 256 [H/8, W/8]
        self.conv17 = vgg16.features[17]
        self.conv19 = vgg16.features[19]
        self.conv21 = vgg16.features[21]  # \\ 512 [H/16, W/16]
        self.conv24 = vgg16.features[24]
        self.conv26 = vgg16.features[26]
        self.conv28 = vgg16.features[28]  # \\ 512 [H/32, W/32]
        self.conv30 = nn.Conv2d(self.conv28.out_channels, 1280, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Feature Decoder
        # self.conv_dec0 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv28.out_channels + self.conv21.out_channels, self.conv21.out_channels, kernel_size=(3, 3),
        #               stride=(1, 1), padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv21.out_channels, self.conv21.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
        #               bias=False),
        # )
        #
        # self.conv_dec1 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv21.out_channels + self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3),
        #               stride=(1, 1), padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
        #               bias=False),
        # )
        #
        # self.conv_dec2 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv14.out_channels + self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3),
        #               stride=(1, 1), padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
        #               bias=False)
        # )
        #
        # self.conv_dec3 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv7.out_channels + self.conv2.out_channels, 32, kernel_size=(3, 3),
        #               stride=(1, 1), padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
        #               bias=False)
        # )
        #
        # self.conv_dec4 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv2.out_channels + self.conv2.out_channels, 32, kernel_size=(3, 3),
        #               stride=(1, 1), padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1,
        #               bias=False)
        # )

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,
                                     return_indices=True)

    def forward(self, x):
        # block0
        x0 = self.conv0(x)
        x1 = self.relu(x0)
        x2 = self.conv2(x1)
        x3, ind3 = self.max_pool(x2)  # [H/2, W/2]

        x4 = self.relu(x3)
        x5 = self.conv5(x4)
        x6 = self.relu(x5)
        x7 = self.conv7(x6)
        x8, ind8 = self.max_pool(x7)  # [H/4, W/4]

        # block2
        x9 = self.relu(x8)
        x10 = self.conv10(x9)
        x11 = self.relu(x10)
        x12 = self.conv12(x11)
        x13 = self.relu(x12)
        x14 = self.conv14(x13)
        x15, ind15 = self.max_pool(x14)  # [H/8, W/8]

        # block3
        x16 = self.relu(x15)
        x17 = self.conv17(x16)
        x18 = self.relu(x17)
        x19 = self.conv19(x18)
        x20 = self.relu(x19)
        x21 = self.conv21(x20)
        x22, ind22 = self.max_pool(x21)  # [H/16, W/16]

        # block4
        x23 = self.relu(x22)
        x24 = self.conv24(x23)
        x25 = self.relu(x24)
        x26 = self.conv26(x25)
        x27 = self.relu(x26)
        x28 = self.conv28(x27)
        x29, ind29 = self.max_pool(x28)  # [H/32, W/32]

        x30 = self.relu(x29)
        x31 = self.conv30(x30)

        if self.level == -1:
            return x31
        elif self.level == 6:
            return x31, [x29, x22, x15, x8, x3]

class VGG2(nn.Module):
    def __init__(self, args, level, estimate_depth=False, attention='none'):
        super(VGG2, self).__init__()
        # print('estimate_depth: ', estimate_depth)
        self.args = args
        self.level = level
        self.estimate_depth = estimate_depth
        self.attention = attention
        # self.enc_feature = []

        vgg16 = torchvision.models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('/ws/external/checkpoints/pretrained/vgg16-397923af.pth'))

        # load CNN from VGG16, the first three block
        # Feature Encoder
        self.conv0 = vgg16.features[0]
        self.conv2 = vgg16.features[2]  # \\64
        self.conv5 = vgg16.features[5]  #
        self.conv7 = vgg16.features[7]  # \\128
        self.conv10 = vgg16.features[10]
        self.conv12 = vgg16.features[12]
        self.conv14 = vgg16.features[14]  # \\ 256
        self.conv17 = vgg16.features[17]
        self.conv19 = vgg16.features[19]
        self.conv21 = vgg16.features[21]  # \\ 512

        # Feature Decoder
        # self.conv_dec0 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv21.out_channels + self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3),
        #               stride=(1, 1), padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
        #               bias=False),
        # )
        #
        # self.conv_dec1 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv14.out_channels + self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3),
        #               stride=(1, 1), padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
        #               bias=False),
        # )
        #
        # self.conv_dec2 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv7.out_channels + self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3),
        #               stride=(1, 1), padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
        #               bias=False)
        # )
        #
        # self.conv_dec3 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.conv2.out_channels + self.conv2.out_channels, 32, kernel_size=(3, 3),
        #               stride=(1, 1), padding=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1,
        #               bias=False)
        # )

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,
                                     return_indices=True)

        # Feature confidence
        if self.attention in ['none', 'v0.0', 'v0.1']:
            self.conf0 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )

            self.conf1 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )
            self.conf2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )
            self.conf3 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )
            self.conf4 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )

        if self.estimate_depth:
            self.depth0 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            )

            self.depth1 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            )
            self.depth2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            )
            self.depth3 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            )
            self.depth4 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            )

            self.depth0.apply(initialize_weights_to_zero)
            self.depth1.apply(initialize_weights_to_zero)
            self.depth2.apply(initialize_weights_to_zero)
            self.depth3.apply(initialize_weights_to_zero)
            self.depth4.apply(initialize_weights_to_zero)

    def forward(self, x):
        # block0
        x0 = self.conv0(x)
        x1 = self.relu(x0)
        x2 = self.conv2(x1)
        x3, ind3 = self.max_pool(x2)  # [H/2, W/2]

        x4 = self.relu(x3)
        x5 = self.conv5(x4)
        x6 = self.relu(x5)
        x7 = self.conv7(x6)
        x8, ind8 = self.max_pool(x7)  # [H/4, W/4]

        # block2
        x9 = self.relu(x8)
        x10 = self.conv10(x9)
        x11 = self.relu(x10)
        x12 = self.conv12(x11)
        x13 = self.relu(x12)
        x14 = self.conv14(x13)
        x15, ind15 = self.max_pool(x14)  # [H/8, W/8]

        # block3
        x16 = self.relu(x15)
        x17 = self.conv17(x16)
        x18 = self.relu(x17)
        x19 = self.conv19(x18)
        x20 = self.relu(x19)
        x21 = self.conv21(x20)
        x22, ind22 = self.max_pool(x21)  # [H/16, W/16]

        self.enc_feature = [x22, x15, x8, x3, x2]

        # # dec0
        # x23 = F.interpolate(x22, [x15.shape[2], x16.shape[3]], mode="nearest")
        # x24 = torch.cat([x23, x15], dim=1)
        # x25 = self.conv_dec0(x24)  # [H/4, W/4]
        #
        # # dec1
        # x26 = F.interpolate(x25, [x8.shape[2], x9.shape[3]], mode="nearest")
        # x27 = torch.cat([x26, x8], dim=1)
        # x28 = self.conv_dec1(x27)  # [H/4, W/4]
        #
        # # dec2
        # x29 = F.interpolate(x28, [x3.shape[2], x3.shape[3]], mode="nearest")
        # x30 = torch.cat([x29, x3], dim=1)
        # x31 = self.conv_dec2(x30)  # [H/2, W/2]
        #
        # # dec3
        # x32 = F.interpolate(x31, [x2.shape[2], x2.shape[3]], mode="nearest")
        # x33 = torch.cat([x32, x2], dim=1)
        # x34 = self.conv_dec3(x33)  # [H, W]

        c0 = nn.Sigmoid()(-self.conf0(x22))
        c1 = nn.Sigmoid()(-self.conf1(x15))
        c2 = nn.Sigmoid()(-self.conf2(x8))
        c3 = nn.Sigmoid()(-self.conf3(x3))
        # c4 = nn.Sigmoid()(-self.conf4(x2))

        if self.estimate_depth:
            d0 = self.depth0(x22)
            d1 = self.depth1(x15)
            d2 = self.depth2(x8)
            d3 = self.depth3(x3)
            # d4 = self.depth4(x2)
        else:
            d0, d1, d2, d3 = None, None, None, None

        if self.level == -1:
            return [x22], [c0], [d0]
        elif self.level == -2:
            return [x15], [c1], [d1]
        elif self.level == -3:
            return [x8], [c2], [d2]
        elif self.level == 2:
            return [x22, x15], [c0, c1], [d0, d1]
        elif self.level == 3:
            return [x22, x15, x8], [c0, c1, c2], [d0, d1, d2]
        elif self.level == 4:
            return [x22, x15, x8, x3], [c0, c1, c2, c3], [d0, d1, d2, d3]


class Conv2dCircularPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w, pad_w, 0, 0], mode='circular')  # horizontal circular padding
            x = F.pad(x, [0, 0, pad_h, pad_h])  # vertical constant padding with zeros
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)

class VGGUnet2_Cycle(nn.Module):
    def __init__(self, args, level, estimate_depth=False, attention='none'):
        super(VGGUnet2_Cycle, self).__init__()
        self.args = args
        self.level = level
        self.estimate_depth = estimate_depth
        self.attention = attention

        vgg16 = torchvision.models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('/ws/external/checkpoints/pretrained/vgg16-397923af.pth'))

        # Load CNN from VGG16, the first three block
        # Feature Encoder
        self.conv0 = Conv2dCircularPadding(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv0.weight.data = vgg16.features[0].weight.data
        self.conv0.bias.data = vgg16.features[0].bias.data

        self.conv2 = Conv2dCircularPadding(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2.weight.data = vgg16.features[2].weight.data
        self.conv2.bias.data = vgg16.features[2].bias.data

        self.conv5 = Conv2dCircularPadding(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv5.weight.data = vgg16.features[5].weight.data
        self.conv5.bias.data = vgg16.features[5].bias.data

        self.conv7 = Conv2dCircularPadding(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv7.weight.data = vgg16.features[7].weight.data
        self.conv7.bias.data = vgg16.features[7].bias.data

        self.conv10 = Conv2dCircularPadding(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv10.weight.data = vgg16.features[10].weight.data
        self.conv10.bias.data = vgg16.features[10].bias.data

        self.conv12 = Conv2dCircularPadding(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv12.weight.data = vgg16.features[12].weight.data
        self.conv12.bias.data = vgg16.features[12].bias.data

        self.conv14 = Conv2dCircularPadding(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv14.weight.data = vgg16.features[14].weight.data
        self.conv14.bias.data = vgg16.features[14].bias.data

        self.conv17 = Conv2dCircularPadding(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv17.weight.data = vgg16.features[17].weight.data
        self.conv17.bias.data = vgg16.features[17].bias.data

        self.conv19 = Conv2dCircularPadding(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv19.weight.data = vgg16.features[19].weight.data
        self.conv19.bias.data = vgg16.features[19].bias.data

        self.conv21 = Conv2dCircularPadding(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv21.weight.data = vgg16.features[21].weight.data
        self.conv21.bias.data = vgg16.features[21].bias.data

        # Feature Decoder
        self.conv_dec0 = nn.Sequential(
            nn.ReLU(inplace=True),
            Conv2dCircularPadding(self.conv21.out_channels + self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            Conv2dCircularPadding(self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        self.conv_dec1 = nn.Sequential(
            nn.ReLU(inplace=True),
            Conv2dCircularPadding(self.conv14.out_channels + self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            Conv2dCircularPadding(self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        self.conv_dec2 = nn.Sequential(
            nn.ReLU(inplace=True),
            Conv2dCircularPadding(self.conv7.out_channels + self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            Conv2dCircularPadding(self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        self.conv_dec3 = nn.Sequential(
            nn.ReLU(inplace=True),
            Conv2dCircularPadding(self.conv2.out_channels + self.conv2.out_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(inplace=True),
            Conv2dCircularPadding(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
        )

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,
                                     return_indices=True)

        if self.attention in ['none', 'v0.0', 'v0.1']:
            self.conf0 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )

            self.conf1 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )
            self.conf2 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )
            self.conf3 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )
            self.conf4 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, x):
        # block0
        x0 = self.conv0(x)
        x1 = F.relu(x0)
        x2 = self.conv2(x1)
        x3, ind3 = self.max_pool(x2)

        x4 = F.relu(x3)
        x5 = self.conv5(x4)
        x6 = F.relu(x5)
        x7 = self.conv7(x6)
        x8, ind8 = self.max_pool(x7)

        # block2
        x9 = F.relu(x8)
        x10 = self.conv10(x9)
        x11 = F.relu(x10)
        x12 = self.conv12(x11)
        x13 = F.relu(x12)
        x14 = self.conv14(x13)
        x15, ind15 = self.max_pool(x14)

        # block3
        x16 = F.relu(x15)
        x17 = self.conv17(x16)
        x18 = F.relu(x17)
        x19 = self.conv19(x18)
        x20 = F.relu(x19)
        x21 = self.conv21(x20)
        x22, ind22 = self.max_pool(x21)

        self.enc_feature = [x22, x15, x8, x3, x2]

        # dec0
        x23 = F.interpolate(x22, [x15.shape[2], x16.shape[3]], mode="nearest")
        x24 = torch.cat([x23, x15], dim=1)
        x25 = self.conv_dec0(x24)  # [H/4, W/4]

        # dec1
        x26 = F.interpolate(x25, [x8.shape[2], x9.shape[3]], mode="nearest")
        x27 = torch.cat([x26, x8], dim=1)
        x28 = self.conv_dec1(x27)  # [H/4, W/4]

        # dec2
        x29 = F.interpolate(x28, [x3.shape[2], x3.shape[3]], mode="nearest")
        x30 = torch.cat([x29, x3], dim=1)
        x31 = self.conv_dec2(x30)  # [H/2, W/2]

        # dec3
        x32 = F.interpolate(x31, [x2.shape[2], x2.shape[3]], mode="nearest")
        x33 = torch.cat([x32, x2], dim=1)
        x34 = self.conv_dec3(x33)  # [H, W]

        c0 = nn.Sigmoid()(-self.conf0(x22))
        c1 = nn.Sigmoid()(-self.conf1(x25))
        c2 = nn.Sigmoid()(-self.conf2(x28))
        c3 = nn.Sigmoid()(-self.conf3(x31))
        c4 = nn.Sigmoid()(-self.conf4(x34))

        d0, d1, d2, d3 = None, None, None, None

        if self.level == -1:
            return [x22], [c0], [d0]
        elif self.level == -2:
            return [x25], [c1], [d1]
        elif self.level == -3:
            return [x28], [c2], [d2]
        elif self.level == 2:
            return [x22, x25], [c0, c1], [d0, d1]
        elif self.level == 3:
            return [x22, x25, x28], [c0, c1, c2], [d0, d1, d2]
        elif self.level == 4:
            return [x22, x25, x28, x31], [c0, c1, c2, c3], [d0, d1, d2, d3]

class VGG2_Cycle(nn.Module):
    def __init__(self, args, level, estimate_depth=False, attention='none'):
        super(VGG2_Cycle, self).__init__()
        self.args = args
        self.level = level
        self.estimate_depth = estimate_depth
        self.attention = attention

        vgg16 = torchvision.models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('/ws/external/checkpoints/pretrained/vgg16-397923af.pth'))

        # Load CNN from VGG16, the first three block
        # Feature Encoder
        self.conv0 = Conv2dCircularPadding(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv0.weight.data = vgg16.features[0].weight.data
        self.conv0.bias.data = vgg16.features[0].bias.data

        self.conv2 = Conv2dCircularPadding(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2.weight.data = vgg16.features[2].weight.data
        self.conv2.bias.data = vgg16.features[2].bias.data

        self.conv5 = Conv2dCircularPadding(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv5.weight.data = vgg16.features[5].weight.data
        self.conv5.bias.data = vgg16.features[5].bias.data

        self.conv7 = Conv2dCircularPadding(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv7.weight.data = vgg16.features[7].weight.data
        self.conv7.bias.data = vgg16.features[7].bias.data

        self.conv10 = Conv2dCircularPadding(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv10.weight.data = vgg16.features[10].weight.data
        self.conv10.bias.data = vgg16.features[10].bias.data

        self.conv12 = Conv2dCircularPadding(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv12.weight.data = vgg16.features[12].weight.data
        self.conv12.bias.data = vgg16.features[12].bias.data

        self.conv14 = Conv2dCircularPadding(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv14.weight.data = vgg16.features[14].weight.data
        self.conv14.bias.data = vgg16.features[14].bias.data

        self.conv17 = Conv2dCircularPadding(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv17.weight.data = vgg16.features[17].weight.data
        self.conv17.bias.data = vgg16.features[17].bias.data

        self.conv19 = Conv2dCircularPadding(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv19.weight.data = vgg16.features[19].weight.data
        self.conv19.bias.data = vgg16.features[19].bias.data

        self.conv21 = Conv2dCircularPadding(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv21.weight.data = vgg16.features[21].weight.data
        self.conv21.bias.data = vgg16.features[21].bias.data

        # Feature Decoder
        # self.conv_dec0 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv21.out_channels + self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        # )
        #
        # self.conv_dec1 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv14.out_channels + self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        # )
        #
        # self.conv_dec2 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv7.out_channels + self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        # )
        #
        # self.conv_dec3 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv2.out_channels + self.conv2.out_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
        # )

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,
                                     return_indices=True)

        if self.attention in ['none', 'v0.0', 'v0.1']:
            self.conf0 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )

            self.conf1 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )
            self.conf2 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )
            self.conf3 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )
            self.conf4 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, x):
        # block0
        x0 = self.conv0(x)
        x1 = F.relu(x0)
        x2 = self.conv2(x1)
        x3, ind3 = self.max_pool(x2)

        x4 = F.relu(x3)
        x5 = self.conv5(x4)
        x6 = F.relu(x5)
        x7 = self.conv7(x6)
        x8, ind8 = self.max_pool(x7)

        # block2
        x9 = F.relu(x8)
        x10 = self.conv10(x9)
        x11 = F.relu(x10)
        x12 = self.conv12(x11)
        x13 = F.relu(x12)
        x14 = self.conv14(x13)
        x15, ind15 = self.max_pool(x14)

        # block3
        x16 = F.relu(x15)
        x17 = self.conv17(x16)
        x18 = F.relu(x17)
        x19 = self.conv19(x18)
        x20 = F.relu(x19)
        x21 = self.conv21(x20)
        x22, ind22 = self.max_pool(x21)

        self.enc_feature = [x22, x15, x8, x3, x2]

        # dec0
        # x23 = F.interpolate(x22, [x15.shape[2], x16.shape[3]], mode="nearest")
        # x24 = torch.cat([x23, x15], dim=1)
        # x25 = self.conv_dec0(x24)  # [H/4, W/4]
        #
        # # dec1
        # x26 = F.interpolate(x25, [x8.shape[2], x9.shape[3]], mode="nearest")
        # x27 = torch.cat([x26, x8], dim=1)
        # x28 = self.conv_dec1(x27)  # [H/4, W/4]
        #
        # # dec2
        # x29 = F.interpolate(x28, [x3.shape[2], x3.shape[3]], mode="nearest")
        # x30 = torch.cat([x29, x3], dim=1)
        # x31 = self.conv_dec2(x30)  # [H/2, W/2]
        #
        # # dec3
        # x32 = F.interpolate(x31, [x2.shape[2], x2.shape[3]], mode="nearest")
        # x33 = torch.cat([x32, x2], dim=1)
        # x34 = self.conv_dec3(x33)  # [H, W]

        c0 = nn.Sigmoid()(-self.conf0(x22))
        c1 = nn.Sigmoid()(-self.conf1(x15))
        c2 = nn.Sigmoid()(-self.conf2(x8))
        c3 = nn.Sigmoid()(-self.conf3(x3))
        # c4 = nn.Sigmoid()(-self.conf4(x2))

        d0, d1, d2, d3 = None, None, None, None

        if self.level == -1:
            return [x22], [c0], [d0]
        elif self.level == -2:
            return [x15], [c1], [d1]
        elif self.level == -3:
            return [x8], [c2], [d2]
        elif self.level == 2:
            return [x22, x15], [c0, c1], [d0, d1]
        elif self.level == 3:
            return [x22, x15, x8], [c0, c1, c2], [d0, d1, d2]
        elif self.level == 4:
            return [x22, x15, x8, x3], [c0, c1, c2, c3], [d0, d1, d2, d3]

class VGG3_Cycle(nn.Module):
    def __init__(self, args, level, estimate_depth=False, attention='none'):
        super(VGG3_Cycle, self).__init__()
        self.args = args
        self.level = level
        self.estimate_depth = estimate_depth
        self.attention = attention

        vgg16 = torchvision.models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('/ws/external/checkpoints/pretrained/vgg16-397923af.pth'))

        # Load CNN from VGG16, the first three block
        # Feature Encoder
        self.conv0 = Conv2dCircularPadding(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv0.weight.data = vgg16.features[0].weight.data
        self.conv0.bias.data = vgg16.features[0].bias.data

        self.conv2 = Conv2dCircularPadding(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2.weight.data = vgg16.features[2].weight.data
        self.conv2.bias.data = vgg16.features[2].bias.data

        self.conv5 = Conv2dCircularPadding(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv5.weight.data = vgg16.features[5].weight.data
        self.conv5.bias.data = vgg16.features[5].bias.data

        self.conv7 = Conv2dCircularPadding(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv7.weight.data = vgg16.features[7].weight.data
        self.conv7.bias.data = vgg16.features[7].bias.data

        self.conv10 = Conv2dCircularPadding(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv10.weight.data = vgg16.features[10].weight.data
        self.conv10.bias.data = vgg16.features[10].bias.data

        self.conv12 = Conv2dCircularPadding(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv12.weight.data = vgg16.features[12].weight.data
        self.conv12.bias.data = vgg16.features[12].bias.data

        self.conv14 = Conv2dCircularPadding(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv14.weight.data = vgg16.features[14].weight.data
        self.conv14.bias.data = vgg16.features[14].bias.data

        self.conv17 = Conv2dCircularPadding(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv17.weight.data = vgg16.features[17].weight.data
        self.conv17.bias.data = vgg16.features[17].bias.data

        self.conv19 = Conv2dCircularPadding(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv19.weight.data = vgg16.features[19].weight.data
        self.conv19.bias.data = vgg16.features[19].bias.data

        self.conv21 = Conv2dCircularPadding(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv21.weight.data = vgg16.features[21].weight.data
        self.conv21.bias.data = vgg16.features[21].bias.data

        self.conv24 = Conv2dCircularPadding(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv24.weight.data = vgg16.features[24].weight.data
        self.conv24.bias.data = vgg16.features[24].bias.data

        self.conv26 = Conv2dCircularPadding(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv26.weight.data = vgg16.features[26].weight.data
        self.conv26.bias.data = vgg16.features[26].bias.data

        self.conv28 = Conv2dCircularPadding(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv28.weight.data = vgg16.features[28].weight.data
        self.conv28.bias.data = vgg16.features[28].bias.data

        self.conv30 = nn.Conv2d(self.conv28.out_channels, 1280, kernel_size=(3, 3), stride=(1, 1), padding=1)

        # Feature Decoder
        # self.conv_dec0 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv21.out_channels + self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        # )
        #
        # self.conv_dec1 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv14.out_channels + self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        # )
        #
        # self.conv_dec2 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv7.out_channels + self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1),
        # )
        #
        # self.conv_dec3 = nn.Sequential(
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(self.conv2.out_channels + self.conv2.out_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
        #     nn.ReLU(inplace=True),
        #     Conv2dCircularPadding(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
        # )

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,
                                     return_indices=True)

        if self.attention in ['none', 'v0.0', 'v0.1']:
            self.conf0 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(512, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )

            self.conf1 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )
            self.conf2 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )
            self.conf3 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )
            self.conf4 = nn.Sequential(
                nn.ReLU(),
                Conv2dCircularPadding(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.Sigmoid(),
            )

    def forward(self, x):
        # block0
        x0 = self.conv0(x)
        x1 = F.relu(x0)
        x2 = self.conv2(x1)
        x3, ind3 = self.max_pool(x2)

        x4 = F.relu(x3)
        x5 = self.conv5(x4)
        x6 = F.relu(x5)
        x7 = self.conv7(x6)
        x8, ind8 = self.max_pool(x7)

        # block2
        x9 = F.relu(x8)
        x10 = self.conv10(x9)
        x11 = F.relu(x10)
        x12 = self.conv12(x11)
        x13 = F.relu(x12)
        x14 = self.conv14(x13)
        x15, ind15 = self.max_pool(x14)

        # block3
        x16 = F.relu(x15)
        x17 = self.conv17(x16)
        x18 = F.relu(x17)
        x19 = self.conv19(x18)
        x20 = F.relu(x19)
        x21 = self.conv21(x20)
        x22, ind22 = self.max_pool(x21)

        # block4
        x23 = self.relu(x22)
        x24 = self.conv24(x23)
        x25 = self.relu(x24)
        x26 = self.conv26(x25)
        x27 = self.relu(x26)
        x28 = self.conv28(x27)
        x29, ind29 = self.max_pool(x28)  # [H/32, W/32]

        x30 = self.relu(x29)
        x31 = self.conv30(x30)

        if self.level == -1:
            return x31
        elif self.level == 6:
            return x31, [x29, x22, x15, x8, x3]

class VGGUnet2_v2(nn.Module):
    def __init__(self, args, level, estimate_depth=False, attention='none'):
        super(VGGUnet2_v2, self).__init__()
        # print('estimate_depth: ', estimate_depth)
        self.args = args
        self.level = level
        self.estimate_depth = estimate_depth
        self.attention = attention
        # self.enc_feature = []

        vgg16 = torchvision.models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('/ws/external/checkpoints/pretrained/vgg16-397923af.pth'))

        # load CNN from VGG16, the first three block
        # Feature Encoder
        self.conv0 = vgg16.features[0]
        self.conv2 = vgg16.features[2]  # \\64
        self.conv5 = vgg16.features[5]  #
        self.conv7 = vgg16.features[7]  # \\128
        self.conv10 = vgg16.features[10]
        self.conv12 = vgg16.features[12]
        self.conv14 = vgg16.features[14]  # \\ 256
        self.conv17 = vgg16.features[17]
        self.conv19 = vgg16.features[19]
        self.conv21 = vgg16.features[21]  # \\ 512

        # Feature Decoder
        self.conv_dec0 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv21.out_channels + self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv14.out_channels, self.conv14.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False),
        )

        self.conv_dec1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv14.out_channels + self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False),
        )

        self.conv_dec2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv7.out_channels + self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False)
        )

        self.conv_dec3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv2.out_channels + self.conv2.out_channels, 32, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False)
        )

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,
                                     return_indices=True)

        # Feature confidence
        if self.attention in ['none', 'v0.0', 'v0.1']:
            self.conf0 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(512, 3, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
            )

            self.conf1 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 3, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
            )
            self.conf2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
            )
            self.conf3 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
            )
            self.conf4 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(16, 3, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
            )

        if self.estimate_depth:
            self.depth0 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            )

            self.depth1 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            )
            self.depth2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            )
            self.depth3 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            )
            self.depth4 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            )

            self.depth0.apply(initialize_weights_to_zero)
            self.depth1.apply(initialize_weights_to_zero)
            self.depth2.apply(initialize_weights_to_zero)
            self.depth3.apply(initialize_weights_to_zero)
            self.depth4.apply(initialize_weights_to_zero)

    def forward(self, x):
        # block0
        x0 = self.conv0(x)
        x1 = self.relu(x0)
        x2 = self.conv2(x1)
        x3, ind3 = self.max_pool(x2)  # [H/2, W/2]

        x4 = self.relu(x3)
        x5 = self.conv5(x4)
        x6 = self.relu(x5)
        x7 = self.conv7(x6)
        x8, ind8 = self.max_pool(x7)  # [H/4, W/4]

        # block2
        x9 = self.relu(x8)
        x10 = self.conv10(x9)
        x11 = self.relu(x10)
        x12 = self.conv12(x11)
        x13 = self.relu(x12)
        x14 = self.conv14(x13)
        x15, ind15 = self.max_pool(x14)  # [H/8, W/8]

        # block3
        x16 = self.relu(x15)
        x17 = self.conv17(x16)
        x18 = self.relu(x17)
        x19 = self.conv19(x18)
        x20 = self.relu(x19)
        x21 = self.conv21(x20)
        x22, ind22 = self.max_pool(x21)  # [H/16, W/16]

        self.enc_feature = [x22, x15, x8, x3, x2]

        # dec0
        x23 = F.interpolate(x22, [x15.shape[2], x16.shape[3]], mode="nearest")
        x24 = torch.cat([x23, x15], dim=1)
        x25 = self.conv_dec0(x24)  # [H/4, W/4]

        # dec1
        x26 = F.interpolate(x25, [x8.shape[2], x9.shape[3]], mode="nearest")
        x27 = torch.cat([x26, x8], dim=1)
        x28 = self.conv_dec1(x27)  # [H/4, W/4]

        # dec2
        x29 = F.interpolate(x28, [x3.shape[2], x3.shape[3]], mode="nearest")
        x30 = torch.cat([x29, x3], dim=1)
        x31 = self.conv_dec2(x30)  # [H/2, W/2]

        # dec3
        x32 = F.interpolate(x31, [x2.shape[2], x2.shape[3]], mode="nearest")
        x33 = torch.cat([x32, x2], dim=1)
        x34 = self.conv_dec3(x33)  # [H, W]

        # c0 = nn.Sigmoid()(-self.conf0(x22))
        # c1 = nn.Sigmoid()(-self.conf1(x25))
        # c2 = nn.Sigmoid()(-self.conf2(x28))
        # c3 = nn.Sigmoid()(-self.conf3(x31))
        # c4 = nn.Sigmoid()(-self.conf4(x34))
        c0 = torch.softmax(self.conf0(x22), dim=1)
        c1 = torch.softmax(self.conf1(x25), dim=1)
        c2 = torch.softmax(self.conf2(x28), dim=1)
        c3 = torch.softmax(self.conf3(x31), dim=1)
        c4 = torch.softmax(self.conf4(x34), dim=1)

        if self.estimate_depth:
            d0 = self.depth0(x22)
            d1 = self.depth1(x25)
            d2 = self.depth2(x28)
            d3 = self.depth3(x31)
            d4 = self.depth4(x34)
        else:
            d0, d1, d2, d3 = None, None, None, None

        if self.level == -1:
            return [x22], [c0], [d0]
        elif self.level == -2:
            return [x25], [c1], [d1]
        elif self.level == -3:
            return [x28], [c2], [d2]
        elif self.level == 2:
            return [x25, x28], [c1, c2], [d1, d2]
        elif self.level == 3:
            return [x22, x25, x28], [c0, c1, c2], [d0, d1, d2]
        elif self.level == 4:
            return [x22, x25, x28, x31], [c0, c1, c2, c3], [d0, d1, d2, d3]


class VGGUnet_G2S(nn.Module):
    def __init__(self, level):
        super(VGGUnet_G2S, self).__init__()
        # print('estimate_depth: ', estimate_depth)

        self.level = level

        vgg16 = torchvision.models.vgg16(pretrained=True)

        # load CNN from VGG16, the first three block
        self.conv0 = vgg16.features[0]
        self.conv2 = vgg16.features[2]  # \\64
        self.conv5 = vgg16.features[5]  #
        self.conv7 = vgg16.features[7]  # \\128
        self.conv10 = vgg16.features[10]
        self.conv12 = vgg16.features[12]
        self.conv14 = vgg16.features[14]  # \\256

        self.conv_dec1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv14.out_channels + self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv7.out_channels, self.conv7.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False),
        )

        self.conv_dec2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv7.out_channels + self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False)
        )

        self.conv_dec3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(self.conv2.out_channels + self.conv2.out_channels, 32, kernel_size=(3, 3),
                      stride=(1, 1), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1,
                      bias=False)
        )

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False,
                                     return_indices=True)

        self.conf0 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.conf1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.conf2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.conf3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # block0
        x0 = self.conv0(x)
        x1 = self.relu(x0)
        x2 = self.conv2(x1)
        x3, ind3 = self.max_pool(x2)  # [H/2, W/2]

        B, C, H2, W2 = x2.shape
        x2_ = x2.reshape(B, C, H2 * 2, W2 // 2)

        B, C, H3, W3 = x3.shape
        x3_ = x3.reshape(B, C, H3*2, W3//2)

        x4 = self.relu(x3)
        x5 = self.conv5(x4)
        x6 = self.relu(x5)
        x7 = self.conv7(x6)
        x8, ind8 = self.max_pool(x7)  # [H/4, W/4]

        B, C, H8, W8 = x8.shape
        x8_ = x8.reshape(B, C, H8 * 2, W8 // 2)

        # block2
        x9 = self.relu(x8)
        x10 = self.conv10(x9)
        x11 = self.relu(x10)
        x12 = self.conv12(x11)
        x13 = self.relu(x12)
        x14 = self.conv14(x13)
        x15, ind15 = self.max_pool(x14)  # [H/8, W/8]

        B, C, H15, W15 = x15.shape
        x15_ = x15.reshape(B, C, H15 * 2, W15 // 2)

        # dec1
        x16 = F.interpolate(x15_, [x8_.shape[2], x8_.shape[3]], mode="nearest")
        x17 = torch.cat([x16, x8_], dim=1)
        x18 = self.conv_dec1(x17)  # [H/4, W/4]

        # dec2
        x19 = F.interpolate(x18, [x3_.shape[2], x3_.shape[3]], mode="nearest")
        x20 = torch.cat([x19, x3_], dim=1)
        x21 = self.conv_dec2(x20)  # [H/2, W/2]

        x22 = F.interpolate(x21, [x2_.shape[2], x2_.shape[3]], mode="nearest")
        x23 = torch.cat([x22, x2_], dim=1)
        x24 = self.conv_dec3(x23)  # [H, W]

        c0 = nn.Sigmoid()(-self.conf0(x15))
        c1 = nn.Sigmoid()(-self.conf1(x18))
        c2 = nn.Sigmoid()(-self.conf2(x21))
        c3 = nn.Sigmoid()(-self.conf3(x24))

        x15 = L2_norm(x15_)
        x18 = L2_norm(x18)
        x21 = L2_norm(x21)
        x24 = L2_norm(x24)

        if self.level == -1:
            return [x15], [c0]
        elif self.level == -2:
            return [x18], [c1]
        elif self.level == -3:
            return [x21], [c2]
        elif self.level == 2:
            return [x18, x21], [c1, c2]
        elif self.level == 3:
            return [x15, x18, x21], [c0, c1, c2]
        elif self.level == 4:
            return [x15, x18, x21, x24], [c0, c1, c2, c3]


def process_depth(d):
    # tanh [-1, 1] to [0, 1.6 or 10]
    B, _, H, W = d.shape
    d = (d + 1)/2
    d1 = torch.cat([d[:, :, :H//2, :] * 10, d[:, :, H//2:, :] * 1.6], dim=2)
    return d1


def process_depthv0_1(d, min_depth=0.1, max_depth=100):
    # d: tanh [-1, 1]
    # upper: -1, lower: min_depth to max_depth
    B, _, H, W = d.shape
    d = (d + 1)/2
    d[:, :, H // 2:, :] = min_depth + (max_depth - min_depth) * d[:, :, H//2:, :]
    d[:, :, :H // 2, :] = -1
    return d


def process_depthv0_1_1(d, min_depth=0, max_depth=10):
    # d: tanh [-1, 1]
    # upper: -1, lower: min_depth to max_depth
    # w + pred_depth
    # pred_depth would be negative value
    B, _, H, W = d.shape
    d = (d - 1)/2
    d[:, :, H // 2:, :] = min_depth + (max_depth - min_depth) * d[:, :, H//2:, :]
    d[:, :, :H // 2, :] = -1
    return d

def process_depthv0_2(d, min_depth=0.1, max_depth=100):
    # tanh [-1, 1] to [0, 1.6 or 10]
    B, _, H, W = d.shape
    # d = (d + 1)/2
    d1 = torch.cat([d[:, :, :H//2, :] * min_depth, d[:, :, H//2:, :] * max_depth], dim=2)
    return d1

# def process_depthv0_2(disp, min_depth=0.1, max_depth=100):
#     min_disp = 1 / max_depth
#     max_disp = 1 / min_depth
#     scaled_disp = min_disp + (max_disp - min_disp) * disp
#     depth = 1 / scaled_disp
#
#     d = 1 / (disp+1e-6)
#     d = min_depth + (max_depth - min_depth) * d
#
#     return depth


def L2_norm(x):
    B, C, H, W = x.shape
    y = F.normalize(x.reshape(B, C*H*W))
    return y.reshape(B, C, H, W)

