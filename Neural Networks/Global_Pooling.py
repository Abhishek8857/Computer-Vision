USE_GPU = True  # Set to True if you have installed tensorflow for GPU

import torch.nn as nn
import torch

class GlobalAvgPool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            pooled = torch.mean(x, dim=(2, 3))
            return pooled


cnn_global_pool = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False, stride=2),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, stride=2),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, stride=2),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    GlobalAvgPool2d(),
    nn.Linear(in_features=64, out_features=10))
    
if USE_GPU:
    cnn_global_pool.cuda()
