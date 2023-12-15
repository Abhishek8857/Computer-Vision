import cv2
import numpy as np
from convinience_functions import convolve_with_two, gauss, gaussdx, map_range
from two_D_histogram import compute_2d_histogram

def compute_maglap_histogram(rgb_im, n_bins):
    # Convert to grayscale
    gray_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2GRAY).astype(float)
    # Compute Gaussian derivatives
    sigma = 2
    kernel_radius = np.ceil(3.0 * sigma)
    x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
    G = gauss(x, sigma)
    D = gaussdx(x, sigma)
    dx = convolve_with_two(gray_im, D, G.T)
    dy = convolve_with_two(gray_im, G, D.T)

    # Compute second derivatives
    dxx = convolve_with_two(dx, D, G.T)
    dyy = convolve_with_two(dy, G, D.T)

    # Compute gradient magnitude and Laplacian
    mag = np.sqrt(dxx**2 + dyy**2)
    lap = dxx + dyy
    mag = map_range(mag, start=0, end=15)
    lap = map_range(lap, start=-5, end=5)
    
    mag_lap = np.stack([mag, lap], axis=-1)

    return mag, lap, compute_2d_histogram(mag_lap, n_bins=16)