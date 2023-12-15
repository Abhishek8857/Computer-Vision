import numpy as np


def compute_1d_histogram(im, n_bins):
    histogram = np.zeros(n_bins)
    height, width = im.shape
    bin_size = 1/n_bins
    for i in range(height):
        for j in range(width):
            index = np.floor(im[i, j]/bin_size).astype(int)
            histogram[index] += 1
    histogram /= np.sum(histogram)

    return histogram
