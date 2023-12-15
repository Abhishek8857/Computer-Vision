import numpy as np
   
    
def gamma_compression(im):
    height, width, layers = im.shape
    compressed_im = np.copy(im)
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                compressed_im[i, j, k] = np.sqrt(compressed_im[i, j, k])
    return compressed_im