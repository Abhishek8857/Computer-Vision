USE_GPU = True  # Set to True if you have installed tensorflow for GPU

import torch.nn as nn

# Training a model using classic softmax regression  
softmax_regression = nn.Sequential(nn.Flatten(), 
                                    nn.Linear(in_features=3072, out_features=10))
if USE_GPU:
    softmax_regression.cuda()
    