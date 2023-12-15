import numpy as np


def nms(scores):
    """Non-maximum suppression"""
    scores_out = np.zeros_like(scores)
    n_rows, n_cols = scores.shape
    indices = []
    # Create a list of indices for a 3x3 Kernel
    for x in range(-1, 2):
        for y in range(-1, 2):
            if x != 0 or y!=0:
                indices.append([x, y])
    # Initialise an empty list
    max_list = []
    
    # Append kernel values and compare with the max from the list
    for i in range(1, n_rows-1):
        for j in range(1, n_cols-1):
            for x, y in indices:
                max_list.append(scores[i+x, j+y])
            max_val = max(max_list)
            if scores[i, j] > max_val:
                scores_out[i, j] = scores[i, j]
            max_list.clear()
    return scores_out