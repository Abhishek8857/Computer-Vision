USE_GPU = True  # Set to True if you have installed tensorflow for GPU

import torch.nn as nn

tanh_mlp = nn.Sequential(
nn.Flatten(),
nn.Linear(in_features=3072, out_features=512),
nn.Tanh(),
nn.Linear(in_features=512, out_features=10))

if USE_GPU:
    tanh_mlp.cuda()