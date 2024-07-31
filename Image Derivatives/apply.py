import numpy as np
from image_derivatives import *

# Read 3 images
image_1 = imread_gray("Computer-Vision/images/Marq_1.jpg")
image_2 = imread_gray("Computer-Vision/images/Marq_2.jpg")
image_3 = imread_gray("Computer-Vision/images/Marq_3.jpg")

# Plot Gauss
x = np.linspace(-5, 5, 100)
y = gaussdx(x, sigma=1.0)
fig, ax = plt.subplots()
ax.plot(x, y)
fig.tight_layout()


# Plot impulse image
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

# Plot first derivative images
grad_dx, grad_dy = gauss_derivs(image_3, sigma=5.0)
plot_multiple(
    [image_3, grad_dx, grad_dy],
    ["Image", "Derivative in x-direction", "Derivative in y-direction"],
)


# Plot second derivative for image 1
grad_dxx, grad_dxy, grad_dyy = gauss_second_derivs(image_1, sigma=2.0)
plot_multiple([image_1, grad_dxx, grad_dxy, grad_dyy], ["Image", "Dxx", "Dxy", "Dyy"])

# Plot second derivative for the image 2
grad_dxx, grad_dxy, grad_dyy = gauss_second_derivs(image_2, sigma=2.0)
plot_multiple([image_2, grad_dxx, grad_dxy, grad_dyy], ["Image", "Dxx", "Dxy", "Dyy"])


# Plot Polar gradients for image 1
grad_mag, grad_dir = image_gradients_polar(image_1, sigma=2.0)
plot_multiple(
    [image_1, grad_mag, grad_dir],
    ["Image", "Magnitude", "Direction"],
    colormap=["gray", "gray", "twilight"],
)

# Plot polar gradients for image 2
grad_mag, grad_theta = image_gradients_polar(image_2, sigma=2.0)
plot_multiple(
    [image_2, grad_mag, grad_theta],
    ["Image", "Magnitude", "Direction"],
    colormap=["gray", "gray", "twilight"],
)


# Plot Laplace gradients for image 1
lap = laplace(image_1, sigma=2.0)
plot_multiple([image_1, lap], ["Image", "Laplace"])

# Plot Laplace gradients for image 2
lap = laplace(image_2, sigma=2.0)
plot_multiple([image_2, lap], ["Image", "Laplace"])

plt.show()
