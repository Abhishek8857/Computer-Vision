from dataclasses import dataclass, field
from typing import Callable

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from one_D_histogram import compute_1d_histogram
from three_D_histogram import compute_3d_histogram
from gradient_histogram import compute_gradient_histogram
from distances import chi_square_distance, euclidean_distance
from maglap_histogram import compute_maglap_histogram
from plot import plot_multiple, plot_3d_histogram


# 1D Histogram Plot 
fig, axes = plt.subplots(1, 4, figsize=(10, 2), constrained_layout=True)
bin_counts = [2, 25, 256]
gray_img = iio.imread("images/terrain.png", mode="L").astype(np.float32) / 256

axes[0].set_title("Image")
axes[0].imshow(gray_img, cmap="gray")
for ax, n_bins in zip(axes[1:], bin_counts):
    ax.set_title(f"1D histogram with {n_bins} bins")
    bin_size = 1 / n_bins
    x_axis = np.linspace(0, 1, n_bins, endpoint=False) + bin_size / 2
    hist = compute_1d_histogram(gray_img, n_bins)
    ax.bar(x_axis, hist, bin_size)
    

# 3D Histogram Plot
paths = ["images/sunset.png", "images/terrain.png"]
images = [iio.imread(p) for p in paths]
plot_multiple(images, paths)

fig, axes = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={"projection": "3d"})
for path, ax in zip(paths, axes):
    im = iio.imread(path).astype(np.float32) / 256
    hist = compute_3d_histogram(im, n_bins=16)  # <--- FIDDLE WITH N_BINS HERE
    plot_3d_histogram(ax, hist, "RGB")
fig.tight_layout()

# Gradient Histogram Plot
paths = ["images/model/obj4__0.png", "images\model\obj42__0.png"]
images, titles = [], []

for path in paths:
    im = iio.imread(path)
    dx, dy, hist = compute_gradient_histogram(im, n_bins=16)
    images += [im, dx, dy, np.log(hist + 1e-3)]
    titles += [path, "dx", "dy", "Histogram (log)"]

plot_multiple(images, titles, max_columns=4, imwidth=2, imheight=2, colormap="viridis")

# Gradient Magnitude and Laplacian Histogram Plot
paths = [f"images/model/obj{i}__0.png" for i in [20, 37, 36, 55]]
images, titles = [], []

for path in paths:
    im = iio.imread(path)
    mag, lap, hist = compute_maglap_histogram(im, n_bins=16)
    images += [im, mag, lap, np.log(hist + 1e-3)]
    titles += [path, "Gradient magn.", "Laplacian", "Histogram (log)"]
plot_multiple(images, titles, imwidth=2, imheight=2, max_columns=4, colormap="viridis")


im1 = iio.imread("images/model/obj1__0.png")
im2 = iio.imread("images/model/obj91__0.png")
im3 = iio.imread("images/model/obj94__0.png")

n_bins = 8
h1 = compute_3d_histogram(im1 / 256, n_bins)
h2 = compute_3d_histogram(im2 / 256, n_bins)
h3 = compute_3d_histogram(im3 / 256, n_bins)

eucl_dist1 = euclidean_distance(h1, h2)
chisq_dist1 = chi_square_distance(h1, h2)
eucl_dist2 = euclidean_distance(h1, h3)
chisq_dist2 = chi_square_distance(h1, h3)

titles = [
    "Reference image",
    f"Eucl: {eucl_dist1:.3f}, ChiSq:  {chisq_dist1:.3f}",
    f"Eucl: {eucl_dist2:.3f}, ChiSq:  {chisq_dist2:.3f}",
]

plot_multiple([im1, im2, im3], titles, imheight=3)
plt.show()
