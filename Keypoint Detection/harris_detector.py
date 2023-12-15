import numpy as np
from convinience_functions import gauss_derivs, gauss_filter


def harris_scores(im, opts):
    dx, dy = gauss_derivs(im, opts.sigma1)
    
    sq_dx = dx ** 2
    sq_dy = dy ** 2
    
    filtered_sq_dx = gauss_filter(sq_dx, opts.sigma2)
    filterered_sq_dy = gauss_filter(sq_dy, opts.sigma2)

    # det(𝑀)=𝜆1𝜆2=(𝐺(𝜎̃ )⋆𝐼2𝑥)⋅(𝐺(𝜎̃ )⋆𝐼2𝑦)−(𝐺(𝜎̃ )⋆(𝐼𝑥⋅𝐼𝑦))^2
    determinant = filtered_sq_dx * filterered_sq_dy - gauss_filter(dx * dy, opts.sigma2)
    # trace(𝑀)=𝜆1+𝜆2=𝐺(𝜎̃ )⋆𝐼2𝑥+𝐺(𝜎̃ )⋆𝐼2𝑦
    trace = filtered_sq_dx + filterered_sq_dy
    # det(𝑀)−𝛼⋅trace2(𝑀)
    scores = determinant - opts.alpha * trace ** 2
    return scores


def score_map_to_keypoints(scores, opts):
    corner_ys, corner_xs = (scores > opts.score_threshold).nonzero()
    return np.stack([corner_xs, corner_ys], axis=1)