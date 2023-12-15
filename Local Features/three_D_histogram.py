import numpy as np

def compute_3d_histogram(im, n_bins):
    histogram = np.zeros([n_bins, n_bins, n_bins], dtype=np.float32)
    bin_size = 1 / n_bins
    height ,width, layers = im.shape
    for i in range(height):
        for j in range(width):
            index = np.floor(im[i, j]/bin_size).astype(int)
            histogram[index[0]][index[1]][index[2]] += 1
    histogram /= np.sum(histogram)
    return histogram