import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os
import glob
import matplotlib.pyplot as plt
import cv2
# import Dataset calss
from torch.utils.data import Dataset



""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = conv_block(in_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        return x


class UNet(nn.Module):
    def __init__(self, in_c, out_c, num_filters=64):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = encoder_block(in_c, num_filters)
        self.encoder2 = encoder_block(num_filters, num_filters * 2)
        self.encoder3 = encoder_block(num_filters * 2, num_filters * 4)
        self.encoder4 = encoder_block(num_filters * 4, num_filters * 8)

        # Bottleneck
        self.bottleneck = conv_block(num_filters * 8, num_filters * 16)

        # Decoder
        self.decoder1 = decoder_block(num_filters * 16, num_filters * 8)
        self.decoder2 = decoder_block(num_filters * 8, num_filters * 4)
        self.decoder3 = decoder_block(num_filters * 4, num_filters * 2)
        self.decoder4 = decoder_block(num_filters * 2, num_filters)

        # Classifier
        self.outputs = nn.Conv2d(num_filters, out_c, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1, x = self.encoder1(x)
        enc2, x = self.encoder2(x)
        enc3, x = self.encoder3(x)
        enc4, x = self.encoder4(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder1(x, enc4)
        x = self.decoder2(x, enc3)
        x = self.decoder3(x, enc2)
        x = self.decoder4(x, enc1)

        # Classifier
        outputs = self.outputs(x)

        return outputs
    

# Inspired by: https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/PyTorch/unet.py
