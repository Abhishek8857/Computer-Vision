import cv2
import imageio.v3 as iio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from convinience_functions import *

# Box/Averaging Filter
def filter_box(image, sigma):
    size = int(round(sigma * 12 ** 0.5))
    kernel = np.array([np.ones(size)])   
    return convolve_with_two(image, kernel, kernel.T)

# Gaussian Filter
def filter_gauss(image, kernel_factor, sigma):
    kernel=gauss(np.arange(start=-kernel_factor*sigma, stop=kernel_factor*sigma+1), sigma)
    kernel = np.array([kernel])
    return convolve_with_two(image, kernel, kernel.T)

# Sampling and Aliasing
def sample_with_gaps(im, period):
    im_result = np.zeros_like(im)
    im_result[::period, ::period] = im[::period, ::period]
    return im_result
    

def sample_without_gaps(im, period):
    im = im[::period, ::period]
    return im


