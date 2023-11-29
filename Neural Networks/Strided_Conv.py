USE_GPU = True  # Set to True if you have installed tensorflow for GPU

import torch.nn as nn

cnn_strides = nn.Sequential(        
    nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False, stride=2),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, stride=2),
    nn.BatchNorm2d(num_features=64),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(in_features=4096, out_features=10))

if USE_GPU:
    cnn_strides.cuda()