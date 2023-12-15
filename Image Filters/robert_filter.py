import numpy as np


def robert_filter(im):
    robert_filt_x = np.array([[0, 1], [-1, 0]])
    robert_filt_y = np.array([[1, 0], [0, -1]])
    height, width, layers = im.shape
    padded_im = np.pad(im, pad_width=((2, 2), (2, 2), (0, 0)), constant_values=0, mode="constant")
    temp_im = np.zeros_like(padded_im)
    final_im = np.zeros_like(padded_im)
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                temp_im[i, j, k] = np.sum(padded_im[i:i+2, j:j+2, k] * robert_filt_x)
    
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                final_im[i, j, k] = np.sum(temp_im[i:i+2, j:j+2, k] * robert_filt_y)
    
    return final_im[:, :, 0]