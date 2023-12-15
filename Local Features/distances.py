import numpy as np

def euclidean_distance(hist1, hist2):
    distances = np.sqrt(np.sum(np.subtract(hist1, hist2)**2))
    return np.round(distances, 3)


def chi_square_distance(hist1, hist2, eps=1e-3):
    num = np.subtract(hist1, hist2)**2
    den = np.add(np.add(hist1, hist2), eps)
    distances = 0.5*np.sum(np.divide(num, den))
    return np.round(distances, 3)