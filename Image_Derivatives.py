import cv2
import imageio.v3 as iio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# Convenience Functions 
def convolve_with_two(image, kernel1, kernel2):
    """Apply two filters, one after the other."""
    image = ndimage.convolve(image, kernel1)
    image = ndimage.convolve(image, kernel2)
    return image


def imread_gray(filename):
    """Read grayscale image."""
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)


def plot_multiple(images, titles, colormap="gray", max_columns=np.inf, share_axes=True):
    """Plot multiple images as subplots on a grid."""
    assert len(images) == len(titles)
    n_images = len(images)
    n_cols = min(max_columns, n_images)
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4, n_rows * 4),
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

    fig.tight_layout()


def gauss(x, sigma):
    return 1.0 / np.sqrt(2.0 * np.pi) / sigma * np.exp(-(x**2) / 2.0 / sigma**2)


# Gaussian Derivative Filter
def gaussdx(x, sigma):
    normalisation_factor = -(1/np.sqrt(2 * np.pi * sigma**3))
    return normalisation_factor * x * np.exp(-(np.square(x)/(2*sigma**2)))

x = np.linspace(-5, 5, 100)
y = gaussdx(x, sigma=1.0)
fig, ax = plt.subplots()
ax.plot(x, y)
fig.tight_layout()

# Make an Impule image
def make_impulse_image(size=25):
    image = np.zeros((size, size), dtype=int)
    image[12, 12] = 255
    return image

# What happens when you apply the following filter combinations on the impulse image?

# first gaussian, then gaussian^T.
# first gaussian, then derivative^T.
# first derivative, then gaussian^T.
# first gaussian^T, then derivative.
# first derivative^T, then gaussian.

# Convinience Functions for kernels
sigma = 6.0
kernel_radius = int(3.0 * sigma)
x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
G = gauss(x, sigma)
D = gaussdx(x, sigma)

impulse = make_impulse_image()
images = [
    impulse,
    convolve_with_two(impulse, G, G.T),
    convolve_with_two(impulse, G, D.T),
    convolve_with_two(impulse, D, G.T),
    convolve_with_two(impulse, G.T, D),
    convolve_with_two(impulse, D.T, G),
]

titles = [
    "original",
    "first G, then G^T",
    "first G, then D^T",
    "first D, then G^T",
    "first G^T, then D",
    "first D^T, then G",
]

plot_multiple(images, titles, max_columns=3)

# gauss_derivs return the 2D Gaussian derivatives of an input image in  𝑥 and  𝑦 direction
def gauss_derivs(image, sigma):
    kernel_radius = int(3.0 * sigma)
    x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
    G = gauss(x, sigma)
    D = gaussdx(x, sigma)
    
    image_dx = convolve_with_two(image, G.T, D)
    image_dy = convolve_with_two(image, D.T, G)

    return image_dx, image_dy

image = imread_gray("images\Marq_3.jpg")
grad_dx, grad_dy = gauss_derivs(image, sigma=5.0)
plot_multiple(
    [image, grad_dx, grad_dy],
    ["Image", "Derivative in x-direction", "Derivative in y-direction"],
)

# gauss_second_derivs hat returns the 2D second Gaussian derivatives
def gauss_second_derivs(image, sigma):
    kernel_radius = int(3.0 * sigma)
    x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
    G = gauss(x, sigma)
    D = gaussdx(x, sigma)
    image_dx, image_dy = gauss_derivs(image, sigma)
    
    image_dxx = convolve_with_two(image_dx, G.T, D)
    image_dxy = convolve_with_two(image_dx, D.T, G)
    image_dyy = convolve_with_two(image_dy, D.T, G)
    
    return image_dxx, image_dxy, image_dyy


# Plot function for image 1
grad_dxx, grad_dxy, grad_dyy = gauss_second_derivs(image, sigma=2.0)
plot_multiple([image, grad_dxx, grad_dxy, grad_dyy], ["Image", "Dxx", "Dxy", "Dyy"])

# Read new image and plot for image 2
image = imread_gray("images\Marq_2.jpg")
grad_dxx, grad_dxy, grad_dyy = gauss_second_derivs(image, sigma=2.0)
plot_multiple([image, grad_dxx, grad_dxy, grad_dyy], ["Image", "Dxx", "Dxy", "Dyy"])

# image_gradients_polar returns two images with the magnitude and orientation 
# of the gradient for each pixel of the input image.
def image_gradients_polar(image, sigma):
    image_dx, image_dy = gauss_derivs(image, sigma)
    magnitude = np.sqrt(image_dx**2 + image_dy**2)
    direction = np.arctan2(image_dx, image_dy)
    return magnitude, direction

image = imread_gray("images\Marq_1.jpg")
grad_mag, grad_dir = image_gradients_polar(image, sigma=2.0)

# Note: the twilight colormap only works since Matplotlib 3.0, use 'gray' in earlier versions.
# Plot function for image 1
plot_multiple(
    [image, grad_mag, grad_dir],
    ["Image", "Magnitude", "Direction"],
    colormap=["gray", "gray", "twilight"],
)

# Read Image 2
image = imread_gray("images\Marq_2.jpg")
grad_mag, grad_theta = image_gradients_polar(image, sigma=2.0)

# Plot function for image 2
plot_multiple(
    [image, grad_mag, grad_theta],
    ["Image", "Magnitude", "Direction"],
    colormap=["gray", "gray", "twilight"],
)

# Laplace returns an image with the Laplacian-of-Gaussian of each pixel if the image
def laplace(image, sigma):
    image_dxx, image_dxy, image_dyy = gauss_second_derivs(image, sigma)
    return image_dxx + image_dyy

# Plot function for image 1
image = imread_gray("images\Marq_1.jpg")
lap = laplace(image, sigma=2.0)
plot_multiple([image, lap], ["Image", "Laplace"])

# Plot function for image 2
image = imread_gray("images\Marq_2.jpg")
lap = laplace(image, sigma=2.0)
plot_multiple([image, lap], ["Image", "Laplace"])

plt.show()
