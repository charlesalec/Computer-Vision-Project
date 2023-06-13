
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F

class Diagonalize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.diag_embed()


class Resize256x256(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.interpolate(x, size=(256, 256), mode='bilinear')

class PreNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_features):
        """
        Initialize GenISP network

        Args:
            in_channels: number of features of the input image
            hidden_channels: list of two numbers which are number of hidden features
            out_features: number of features in output layer
        """
        super(PreNet, self).__init__()        
        self.convWB = self.conv_wb(in_channels, hidden_channels, 3)
        self.convCC = self.conv_cc(in_channels, hidden_channels, 9)
        self.shallow = self.shallowSQ(in_channels,hidden_channels,out_features)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

    def forward(self, batch_input):
        N, C, H, W = batch_input.shape              # Save the old dimensions
        
        # Create the WhiteBalance correction matrix from the sub-network and apply it to the (non-resized) image(s)
        whitebalance = self.convWB(batch_input)
        batch_input = torch.bmm(whitebalance, batch_input.view(N, C, H*W)).view(N, C, H, W)

        # Create the ColorCorrection matrix from the sub-network and apply it to the (non-resized) image(s)
        colorcorrection = self.convCC(batch_input)
        batch_input = torch.bmm(colorcorrection, batch_input.view(N, C, H*W)).view(N, C, H, W)
        return self.shallow(batch_input)

    def conv_wb(self,in_channels, hidden_channels, out_features):
        return nn.Sequential(
            Resize256x256(),
            nn.Conv2d(in_channels, hidden_channels[0],
                                kernel_size=7),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), #kernel size set to 2 (paper doesn't specify any size)
            nn.Conv2d(hidden_channels[0], hidden_channels[1],
                                kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels[1], hidden_channels[2],
                                kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(hidden_channels[2], out_features),
            Diagonalize())
        
    def conv_cc(self,in_channels, hidden_channels, out_features):
        return nn.Sequential(
            Resize256x256(),
            nn.Conv2d(in_channels, hidden_channels[0],
                                kernel_size=7),
            nn.LeakyReLU(),
            nn.MaxPool2d(2), #kernel size set to 2 (paper doesn't specify any size)
            nn.Conv2d(hidden_channels[0], hidden_channels[1],
                                kernel_size=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels[1], hidden_channels[2],
                                kernel_size=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(hidden_channels[2], out_features),
            nn.Unflatten(1, (3, 3)))

    def shallowSQ(self,in_channels, hidden_channels, out_features):
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels[3],
                                kernel_size=3),
            nn.InstanceNorm2d(hidden_channels[3], affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_channels[3], hidden_channels[4],
                                kernel_size=3),
            nn.InstanceNorm2d(hidden_channels[4], affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_channels[4], hidden_channels[5],
                                kernel_size=1))
