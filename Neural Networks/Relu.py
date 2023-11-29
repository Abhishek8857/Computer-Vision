USE_GPU = True  # Set to True if you have installed tensorflow for GPU

import torch.nn as nn

relu_mlp = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=3072, out_features=512),
    nn.ReLU(),
    nn.Linear(in_features=512, out_features=10)
    )
    
if USE_GPU:
    relu_mlp.cuda()
