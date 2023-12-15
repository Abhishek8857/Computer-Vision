from gamma_comp import gamma_compression  
import numpy as np  


def compute_gradients(im):
    comp_im = gamma_compression(im)
    filt = np.array([[-1, 0, 1]])
    padded_im = np.pad(comp_im, pad_width=((3, 3), (3, 3), (0, 0)), constant_values=0, mode="constant")
    height, width, layers = im.shape
    temp_im = np.zeros_like(padded_im)
    grad_im = np.zeros_like(padded_im)
    
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                temp_im[i, j, k] = np.sum(padded_im[i, j:j+3, k] * filt)
                
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                grad_im[i, j, k] = np.sum(temp_im[i:i+3, j, k] * filt)
                
    return grad_im[:, :, 0]