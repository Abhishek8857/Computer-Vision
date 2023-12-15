import numpy as np
from convinience_functions import gauss_second_derivs


def hessian_scores(im, opts):
    sigma = opts.sigma1
    dxx, dxy, dyy = gauss_second_derivs(im, sigma=sigma)
    scores = (sigma ** 4) * (dxx*dyy - np.square(dxy))

    return scores 

def score_map_to_keypoints(scores, opts):
    corner_ys, corner_xs = (scores > opts.score_threshold).nonzero()
    return np.stack([corner_xs, corner_ys], axis=1)