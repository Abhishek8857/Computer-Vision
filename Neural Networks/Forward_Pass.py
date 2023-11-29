USE_GPU = True  # Set to True if you have installed tensorflow for GPU

import numpy as np
from CNN import *
from Neural_Networks import train_data

def conv3x3_same(x, weights, biases):
    """Convolutional layer with filter size 3x3 and 'same' padding.
    `x` is a NumPy array of shape [in_channels, height, width]
    `weights` has shape [out_channels, in_channels, kernel_height, kernel_width]
    `biases` has shape [out_channels]
    Return the output of the 3x3 conv (without activation)
    """
    
    im_layers, im_height, im_width = x.shape
    kl_height, kl_width = weights.shape[2], weights.shape[3]
    in_chan, out_chan = x.shape[0], biases.shape[0]
    
    assert in_chan == weights.shape[1]
    
    result = np.zeros((out_chan, im_height, im_width))

    # data = (3, 32, 32)
    # weights = (64, 3, 3, 3)
    # biases = (64)
    pad_size = int((kl_height - 1) / 2)
    x_pad = np.pad(x, 
                pad_width=((0, 0), 
                        (pad_size, pad_size), 
                        (pad_size, pad_size)),
                mode='constant')
    
    for i in range(out_chan):
        for j in range(im_height):
            for k in range(im_width):
                result[i, j, k] = np.sum(x_pad[:, j:j + 3, k:k + 3] * weights[i, :, :, :]) + biases[i]

        
    return result       


def maxpool2x2(x):
    """Max pooling with pool size 2x2 and stride 2.
    `x` is a numpy array of shape [in_channels, height, width]
    """
    pool_size = 2
    in_chan, im_height, im_width = x.shape
    pooled_out = int(((im_height - pool_size) / 2) + 1)
    result = np.zeros((in_chan, pooled_out, pooled_out))
    
    for i in range(in_chan):
        for j in range(pooled_out):
            for k in range(pooled_out):
                result[i, j, k] = np.max(x[i, j * pool_size: (j * pool_size) + pool_size, k * pool_size:(k * pool_size) + pool_size])
    return result


def linear(x, weights, biases):
    result = weights @ x + biases
    return result
    
    
def relu(x):
    result = np.maximum(0, x)
    return result


def my_predict_cnn(x, W1, b1, W2, b2, W3, b3):
    x = conv3x3_same(x, W1, b1)
    x = relu(x)
    x = maxpool2x2(x)
    x = conv3x3_same(x, W2, b2)
    x = relu(x)
    x = maxpool2x2(x)
    x = x.reshape(-1)
    x = linear(x, W3, b3)
    return x


W1 = cnn[0].weight.data.cpu().numpy()
b1 = cnn[0].bias.data.cpu().numpy()
W2 = cnn[3].weight.data.cpu().numpy()
b2 = cnn[3].bias.data.cpu().numpy()
W3 = cnn[7].weight.data.cpu().numpy()
b3 = cnn[7].bias.data.cpu().numpy()

inp = train_data[0][0]
inp_np = inp.numpy()
if USE_GPU:
    inp = inp.cuda()