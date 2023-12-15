import numpy as np


def prewitt_filter(im):
    prewitt_filt_x = np.array([[-1, 0 , 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_filt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    height, width, layers = im.shape
    padded_im = np.pad(im, pad_width=((3, 3), (3, 3), (0, 0)), constant_values=0, mode="constant")
    temp_im = np.zeros_like(padded_im)
    final_im = np.zeros_like(padded_im)
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                temp_im[i, j, k] = np.sum(padded_im[i:i+3, j:j+3, k] * prewitt_filt_x)
    
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                final_im[i, j, k] = np.sum(temp_im[i:i+3, j:j+3, k] * prewitt_filt_y)
    
    return final_im[:, :, 0]