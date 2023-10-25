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


fig, axes = plt.subplots(1, 4, figsize=(10, 2), constrained_layout=True)
bin_counts = [2, 25, 256]
gray_img = iio.imread("images\Cat.jpg", mode="L").astype(np.float32) / 256

axes[0].set_title("Image")
axes[0].imshow(gray_img, cmap="gray")
for ax, n_bins in zip(axes[1:], bin_counts):
    ax.set_title(f"1D histogram with {n_bins} bins")
    bin_size = 1 / n_bins
    x_axis = np.linspace(0, 1, n_bins, endpoint=False) + bin_size / 2
    hist = compute_1d_histogram(gray_img, n_bins)
    ax.bar(x_axis, hist, bin_size)
    
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

def plot_3d_histogram(ax, data, axis_names="xyz"):
    """Plot a 3D histogram. We plot a sphere for each bin,
    with volume proportional to the bin content."""
    r, g, b = np.meshgrid(
        *[np.linspace(0, 1, dim) for dim in data.shape], indexing="ij"
    )
    colors = np.stack([r, g, b], axis=-1).reshape(-1, 3)
    marker_sizes = 300 * data ** (1 / 3)
    ax.scatter(r.flat, g.flat, b.flat, s=marker_sizes.flat, c=colors, alpha=0.5)
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    ax.set_zlabel(axis_names[2])

paths = ["images\Marq_2.jpg", "images\Quart_1.jpg"]
images = [iio.imread(p) for p in paths]
plot_multiple(images, paths)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={"projection": "3d"})
for path, ax in zip(paths, axes):
    im = iio.imread(path).astype(np.float32) / 256
    hist = compute_3d_histogram(im, n_bins=16)  # <--- FIDDLE WITH N_BINS HERE
    plot_3d_histogram(ax, hist, "RGB")
fig.tight_layout()

plt.show()
