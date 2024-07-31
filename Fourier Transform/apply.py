from fourier_transform import *


# Plot the image
im_1 = imread_gray("Computer-Vision/images/grass.jpeg")
im_2 = imread_gray("Computer-Vision/images/zebra.jpg")
im_pattern = generate_pattern()
plot_with_spectra(
    [im_1, im_2, im_pattern], ["Image 1", "Image 2", "Pattern image"]
)

# Plot the image
sigma = 3
im = im_1

gauss_filtered = filter_gauss(im, kernel_factor=6, sigma=sigma)
box_filtered = filter_box(im, sigma)
plot_with_spectra(
    [im, box_filtered, gauss_filtered], ["Image", "Box filtered", "Gauss filtered"]
)


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
