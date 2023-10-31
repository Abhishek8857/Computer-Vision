from dataclasses import dataclass, field
from typing import Callable

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# Many useful functions

def plot_multiple(
    images,
    titles=None,
    colormap="gray",
    max_columns=np.inf,
    imwidth=4,
    imheight=4,
    share_axes=False,
):
    """Plot multiple images as subplots on a grid."""
    if titles is None:
        titles = [""] * len(images)
    assert len(images) == len(titles)
    n_images = len(images)
    n_cols = min(max_columns, n_images)
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * imwidth, n_rows * imheight),
        squeeze=False,
        sharex=share_axes,
        sharey=share_axes,
    )

    axes = axes.flat
    # Hide subplots without content
    for ax in axes[n_images:]:
        ax.axis("off")

    if not isinstance(colormap, (list, tuple)):
        colormaps = [colormap] * n_images
    else:
        colormaps = colormap

    for ax, image, title, cmap in zip(axes, images, titles, colormaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.tight_layout()


def load_image(f_name):
    return iio.imread(f_name, mode="L").astype(np.float32) / 255


def convolve_with_two(image, kernel1, kernel2):
    """Apply two filters, one after the other."""
    image = ndimage.convolve(image, kernel1)
    image = ndimage.convolve(image, kernel2)
    return image


def gauss(x, sigma):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x**2) / 2 / sigma**2)


def gaussdx(x, sigma):
    return -1 / np.sqrt(2 * np.pi) / sigma**3 * x * np.exp(-(x**2) / 2 / sigma**2)


def gauss_derivs(image, sigma):
    kernel_radius = np.ceil(3.0 * sigma)
    x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
    G = gauss(x, sigma)
    D = gaussdx(x, sigma)
    image_dx = convolve_with_two(image, D, G.T)
    image_dy = convolve_with_two(image, G, D.T)
    return image_dx, image_dy


def gauss_filter(image, sigma):
    kernel_radius = np.ceil(3.0 * sigma)
    x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
    G = gauss(x, sigma)
    return convolve_with_two(image, G, G.T)


def gauss_second_derivs(image, sigma):
    kernel_radius = np.ceil(3.0 * sigma)
    x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
    G = gauss(x, sigma)
    D = gaussdx(x, sigma)

    image_dx, image_dy = gauss_derivs(image, sigma)
    image_dxx = convolve_with_two(image_dx, D, G.T)
    image_dyy = convolve_with_two(image_dy, G, D.T)
    image_dxy = convolve_with_two(image_dx, G, D.T)
    return image_dxx, image_dxy, image_dyy


def map_range(x, start, end):
    """Maps values `x` that are within the range [start, end) to the range [0, 1)
    Values smaller than `start` become 0, values larger than `end` become
    slightly smaller than 1."""
    return np.clip((x - start) / (end - start), 0, 1 - 1e-10)


def draw_keypoints(image, points):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    radius = image.shape[1] // 100 + 1
    for x, y in points:
        cv2.circle(image, (int(x), int(y)), radius, (1, 0, 0), thickness=2)
    return image


def draw_point_matches(im1, im2, point_matches):
    result = np.concatenate([im1, im2], axis=1)
    result = (result.astype(float) * 0.6).astype(np.uint8)
    im1_width = im1.shape[1]
    for x1, y1, x2, y2 in point_matches:
        cv2.line(
            result,
            (x1, y1),
            (im1_width + x2, y2),
            color=(0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return result


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


def compute_2d_histogram(im, n_bins):
    histogram = np.zeros([n_bins, n_bins], dtype=np.float32)
    bin_size = 1 / n_bins
    height, width, layers = im.shape
    for i in range(height):
        for j in range(width):
            index = np.floor(im[i, j] / bin_size).astype(int)
            histogram[index[0]][index[1]] += 1
    histogram /= np.sum(histogram)
    return histogram

# --------------------------------------------------------------------------------

# Determine Harris Scores
 

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


def score_map_to_keypoints(scores, opts):
    corner_ys, corner_xs = (scores > opts.score_threshold).nonzero()
    return np.stack([corner_xs, corner_ys], axis=1)


class HarrisOpts:
    sigma1: float = 2
    sigma2: float = 2 * 2
    alpha: float = 0.06
    score_threshold: float = 1e-8


opts = HarrisOpts()


paths = ["images/checkboard.jpg", "images/graf.png", "images/gantrycrane.png"]
images = []
titles = []
for path in paths:
    image = load_image(path)

    score_map = harris_scores(image, opts)
    score_map_nms = nms(score_map)
    keypoints = score_map_to_keypoints(score_map_nms, opts)
    keypoint_image = draw_keypoints(image, keypoints)

    images += [score_map, keypoint_image]
    titles += ["Harris response scores", "Harris keypoints"]
plot_multiple(images, titles, max_columns=2, colormap="viridis")


def hessian_scores(im, opts):
    sigma = opts.sigma1
    dxx, dxy, dyy = gauss_second_derivs(im, sigma=sigma)
    scores = (sigma ** 4) * (dxx*dyy - np.square(dxy))

    return scores 


class HessianOpts:
    sigma1: float = 3
    score_threshold: float = 5e-4


opts = HessianOpts()

paths = ["images/checkboard.jpg", "images/graf.png", "images/gantrycrane.png"]

images = []
titles = []
for path in paths:
    image = load_image(path)
    score_map = hessian_scores(image, opts)
    score_map_nms = nms(score_map)
    keypoints = score_map_to_keypoints(score_map_nms, opts)
    keypoint_image = draw_keypoints(image, keypoints)
    images += [score_map, keypoint_image]
    titles += ["Hessian scores", "Hessian keypoints"]

plot_multiple(images, titles, max_columns=2, colormap="viridis")
plt.show()