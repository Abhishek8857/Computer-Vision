import cv2
import imageio.v3 as iio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from plot import plot_multiple


# Define a Gaussian Function
def gauss(x, sigma):
    return (1/(np.sqrt(2 * np.pi) * sigma) * np.exp(-(np.square(x)/(2 * sigma ** 2))))


# Plot the Gaussian for visualization
x = np.linspace(-5, 5, 100)
y = gauss(x, sigma=2)
fig, ax = plt.subplots()
ax.plot(x, y)
fig.tight_layout()


# Define the Gaussian filtering operation
def gaussian_filter(image, sigma, padding=True):
    size = (6 * sigma) + 1  
    half_size = 3*sigma
    g_filter = gauss(np.arange(start=-half_size, stop=half_size + 1, dtype=float), sigma)

    # Valid pad the image
    padded_im = np.pad(image, 
                       pad_width=((half_size, half_size), (half_size, half_size), (0, 0)),
                       constant_values=0, 
                       mode='constant')
    
    height, width, layers = image.shape
    image_temp = np.zeros_like(padded_im)
    image_result = np.zeros_like(image)
    
    # 1-D multiplication through rows
    for i in range(height):
        for j in range(width):
            for k in range(layers):
                image_temp[i+half_size, j+half_size, k] = np.sum(padded_im[i+half_size, j:j+size, k]*g_filter)      
                
    # 1-D Multiplication through columns
    for i in range(height):
        for j in range(width):  
            for k in range(layers):
                image_result[i, j, k] = np.sum(image_temp[i:i+size, j+half_size, k]*g_filter)
    
    return image_result


# Verify the implementation with cv2
def gauss_cv(image, sigma):
    ks = 2 * int(np.ceil(3 * sigma)) + 1
    return cv2.GaussianBlur(image, (ks, ks), sigma, cv2.BORDER_DEFAULT)

# Find the difference between the implemented and cv2 blurred images
def abs_diff(image1, image2):
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    return np.mean(np.abs(image1 - image2), axis=-1)


