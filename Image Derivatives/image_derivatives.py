import cv2
import imageio.v3 as iio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from convinience_functions import *


# Gauss Function
def gauss(x, sigma):
    return 1.0 / np.sqrt(2.0 * np.pi) / sigma * np.exp(-(x**2) / 2.0 / sigma**2)


# Gaussian Derivative Filter
def gaussdx(x, sigma):
    normalisation_factor = -(1/np.sqrt(2 * np.pi * sigma**3))
    return normalisation_factor * x * np.exp(-(np.square(x)/(2*sigma**2)))



# Make an Impule image
def make_impulse_image(size=25):
    image = np.zeros((size, size), dtype=int)
    image[12, 12] = 255
    return image

# What happens when you apply the following filter combinations on the impulse image?

# first gaussian, then gaussian^T.
# first gaussian, then derivative^T.
# first derivative, then gaussian^T.
# first gaussian^T, then derivative.
# first derivative^T, then gaussian.


# gauss_derivs return the 2D Gaussian derivatives of an input image in  𝑥 and  𝑦 direction
def gauss_derivs(image, sigma):
    kernel_radius = int(3.0 * sigma)
    x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
    G = gauss(x, sigma)
    D = gaussdx(x, sigma)
    
    image_dx = convolve_with_two(image, G.T, D)
    image_dy = convolve_with_two(image, D.T, G)

    return image_dx, image_dy


# gauss_second_derivs hat returns the 2D second Gaussian derivatives
def gauss_second_derivs(image, sigma):
    kernel_radius = int(3.0 * sigma)
    x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
    G = gauss(x, sigma)
    D = gaussdx(x, sigma)
    image_dx, image_dy = gauss_derivs(image, sigma)
    
    image_dxx = convolve_with_two(image_dx, G.T, D)
    image_dxy = convolve_with_two(image_dx, D.T, G)
    image_dyy = convolve_with_two(image_dy, D.T, G)
    
    return image_dxx, image_dxy, image_dyy


# image_gradients_polar returns two images with the magnitude and orientation 
# of the gradient for each pixel of the input image.
def image_gradients_polar(image, sigma):
    image_dx, image_dy = gauss_derivs(image, sigma)
    magnitude = np.sqrt(image_dx**2 + image_dy**2)
    direction = np.arctan2(image_dx, image_dy)
    return magnitude, direction



# Note: the twilight colormap only works since Matplotlib 3.0, use 'gray' in earlier versions.

# Laplace returns an image with the Laplacian-of-Gaussian of each pixel if the image
def laplace(image, sigma):
    image_dxx, image_dxy, image_dyy = gauss_second_derivs(image, sigma)
    return image_dxx + image_dyy


