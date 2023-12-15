import cv2
import numpy as np
from convinience_functions import map_range, gauss_derivs
from two_D_histogram import compute_2d_histogram


def compute_gradient_histogram(rgb_im, n_bins):
    # Convert to grayscale
    gray_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2GRAY).astype(float)
    # Compute Gaussian derivatives
    dx, dy = gauss_derivs(gray_im, sigma=2.0)
    # Map the derivatives between -10 and 10 to be between 0 and 1
    dx = map_range(dx, start=-10, end=10)
    dy = map_range(dy, start=-10, end=10)
    # Stack the two derivative images along a new
    # axis at the end (-1 means "last")
    gradients = np.stack([dy, dx], axis=-1)
    return dx, dy, compute_2d_histogram(gradients, n_bins=16)