import numpy as np
import cv2
import imageio.v2 as iio
import matplotlib.pyplot as plt


def median_filter(image):
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
                temp.sort()
                image[i, j, k] = temp[np.floor(len(temp)/2).astype(int)]
    return image


