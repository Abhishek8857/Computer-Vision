USE_GPU = True  # Set to True if you have installed tensorflow for GPU

import torch.nn as nn
import torch
from Global_Pooling import GlobalAvgPool2d
import torch.nn.functional as F

class ResNetBlock(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            self.activation = nn.ReLU()
            # The shortcut connection is just the identity. If feature
            # channel counts differ between input and output, zero
            # padding is used to match the depths. This is implemented
            # by a convolution with the following fixed weight:
            self.pad_weight = nn.Parameter(
                torch.eye(out_channels, in_channels)[:, :, None, None], requires_grad=False
            )
            self.stride = stride

        def forward(self, x):
            r = self.f(x)
            # We apply the padding weight using torch.functional.conv2d
            # which allows us to use a custom weight matrix.
            x = F.conv2d(x, self.pad_weight, stride=self.stride)
            return self.activation(x + r)


class ResNet(torch.nn.Module):
    def __init__(self, num_layers=14, in_channels=3, out_features=10):
        super().__init__()
        if (num_layers - 2) % 6 != 0:
            raise ValueError("n_layers should be 6n+2 (eg 20, 32, 44, 56)")
        n = (num_layers - 2) // 6

        layers = []

        first_layer = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        layers.append(first_layer)

        for i in range(n):
            layers.append(ResNetBlock(16, 16))

        layers.append(ResNetBlock(16, 32, stride=2))
        for i in range(1, n):
            layers.append(ResNetBlock(32, 32))

        layers.append(ResNetBlock(32, 64, stride=2))
        for i in range(1, n):
            layers.append(ResNetBlock(64, 64))

        layers.append(GlobalAvgPool2d())
        layers.append(nn.Linear(64, out_features))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
resnet = ResNet()
if USE_GPU:
    resnet.cuda()