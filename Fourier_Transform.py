import cv2
import imageio.v3 as iio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


# Convinience Functions

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


def imread_gray(filename):
    """Read grayscale image from our data directory."""
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.float32)


def convolve_with_two(image, kernel1, kernel2):
    """Apply two filters, one after the other."""
    image = ndimage.convolve(image, kernel1, mode="wrap")
    image = ndimage.convolve(image, kernel2, mode="wrap")
    return image


def fourier_spectrum(im):
    normalized_im = im / np.sum(im)
    f = np.fft.fft2(normalized_im)
    return np.fft.fftshift(f)


def log_magnitude_spectrum(im):
    return np.log(np.abs(fourier_spectrum(im)) + 1e-8)


def plot_with_spectra(images, titles):
    """Plots a list of images in the first column and the logarithm of their
    magnitude spectrum in the second column."""

    assert len(images) == len(titles)
    n_cols = 2
    n_rows = len(images)
    fig, axes = plt.subplots(n_rows, 2, figsize=(n_cols * 4, n_rows * 4), squeeze=False)

    spectra = [log_magnitude_spectrum(im) for im in images]

    lower = min(np.percentile(s, 0.1) for s in spectra)
    upper = min(np.percentile(s, 99.999) for s in spectra)
    normalizer = mpl.colors.Normalize(vmin=lower, vmax=upper)

    for ax, image, spectrum, title in zip(axes, images, spectra, titles):
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title(title)
        ax[0].set_axis_off()
        c = ax[1].imshow(spectrum, norm=normalizer, cmap="viridis")
        ax[1].set_title("Log magnitude spectrum")
        ax[1].set_axis_off()

    fig.tight_layout()


def gauss(x, sigma):
    return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x**2) / 2 / sigma**2)


def generate_pattern():
    x = np.linspace(0, 1, 256, endpoint=False)
    y = np.sin(x**2 * 16 * np.pi)
    return np.outer(y, y) / 2 + 0.5

# Plot the image
im_1 = imread_gray("images\grass.jpeg")
im_2 = imread_gray("images\zebra.jpg")
im_pattern = generate_pattern()
plot_with_spectra(
    [im_1, im_2, im_pattern], ["Image 1", "Image 2", "Pattern image"]
)

# Box/Averaging Filter
def filter_box(image, sigma):
    size = int(round(sigma * 12 ** 0.5))
    kernel = np.array([np.ones(size)])   
    return convolve_with_two(image, kernel, kernel.T)

# Gaussian Filter
def filter_gauss(image, kernel_factor, sigma):
    kernel=gauss(np.arange(start=-kernel_factor*sigma, stop=kernel_factor*sigma+1), sigma)
    kernel = np.array([kernel])
    return convolve_with_two(image, kernel, kernel.T)

# Plot the image
sigma = 3
im = im_1

gauss_filtered = filter_gauss(im, kernel_factor=6, sigma=sigma)
box_filtered = filter_box(im, sigma)
plot_with_spectra(
    [im, box_filtered, gauss_filtered], ["Image", "Box filtered", "Gauss filtered"]
)

# Sampling and Aliasing

def sample_with_gaps(im, period):
    im_result = np.zeros_like(im)
    im_result[::period, ::period] = im[::period, ::period]
    return im_result
    

def sample_without_gaps(im, period):
    im = im[::period, ::period]
    return im

# Plot for a period of 4
N = 4
im = im_1
sampled_gaps = sample_with_gaps(im, N)
sampled = sample_without_gaps(im, N)

blurred = filter_gauss(im, kernel_factor=6, sigma=4)
blurred_and_sampled_gaps = sample_with_gaps(blurred, N)
blurred_and_sampled = sample_without_gaps(blurred, N)

plot_with_spectra(
    [im, sampled_gaps, sampled, blurred, blurred_and_sampled_gaps, blurred_and_sampled],
    [
        "Original",
        "Sampled (w/ gaps)",
        "Sampled",
        "Gauss blurred",
        "Blurred and s. (w/ gaps)",
        "Blurred and s.",
    ],
)

# Plot for a period of 16
N = 16
image = im_pattern
downsampled_gaps = sample_with_gaps(im_pattern, N)
downsampled = sample_without_gaps(im_pattern, N)

blurred = filter_gauss(image, kernel_factor=6, sigma=12)
blurred_and_downsampled_gaps = sample_with_gaps(blurred, N)
blurred_and_downsampled = sample_without_gaps(blurred, N)

plot_with_spectra(
    [
        im_pattern,
        downsampled_gaps,
        downsampled,
        blurred,
        blurred_and_downsampled_gaps,
        blurred_and_downsampled,
    ],
    [
        "Original",
        "Downsampled (w/ gaps)",
        "Downsampled (no gaps)",
        "Gauss blurred",
        "Blurred and ds. (w/ gaps)",
        "Blurred and downs. (no gaps)",
    ],
)

plt.show()

