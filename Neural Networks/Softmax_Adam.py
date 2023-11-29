USE_GPU = True  # Set to True if you have installed tensorflow for GPU

import torch.nn as nn

softmax_regression_adam = nn.Sequential(nn.Flatten(), 
                                        nn.Linear(in_features=3072, out_features=10))
if USE_GPU:
    softmax_regression_adam.cuda()