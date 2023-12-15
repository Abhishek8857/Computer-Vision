import numpy as np
import imageio.v2 as iio
from plot import plot_multiple

    
def average_filter(image):
    assert len(image.shape) == 3
    height, width, layers = image.shape
    image = np.pad(image, pad_width=((1, 1), (1, 1), (0, 0)), 
                   constant_values=0, 
                   mode="constant")
    for i in range(height):
        for j in range(width):
            for k in range(layers-1):
                temp = []
                for l in range(-1, 2):
                    for m in range(-1, 2):
                        temp.append(image[i+l, j+m, k])
                image[i, j, k] = np.divide(sum(temp), len(temp)).astype(int)
    return image



