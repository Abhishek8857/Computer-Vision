import numpy as np
from convinience_functions import *

# Gauss function
def gauss(x, sigma):
    return 1.0 / np.sqrt(2.0 * np.pi) / sigma * np.exp(-(x**2) / 2.0 / sigma**2)


# Gaussian Derivative Filter
def gaussdx(x, sigma):
    normalisation_factor = -(1/np.sqrt(2 * np.pi * sigma**3))
    return normalisation_factor * x * np.exp(-(np.square(x)/(2*sigma**2)))


# Convinience Functions for kernels
sigma = 6.0
kernel_radius = int(3.0 * sigma)
x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
G = gauss(x, sigma)
D = gaussdx(x, sigma)


# gauss_derivs return the 2D Gaussian derivatives of an input image in  𝑥 and  𝑦 direction
def gauss_derivs(image, sigma):
    kernel_radius = int(3.0 * sigma)
    x = np.arange(-kernel_radius, kernel_radius + 1)[np.newaxis]
    G = gauss(x, sigma)
    D = gaussdx(x, sigma)
    
    image_dx = convolve_with_two(image, G.T, D)
    image_dy = convolve_with_two(image, D.T, G)

    return image_dx, image_dy

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


# image_gradients_polar returns two images with the magnitude and orientation 
# of the gradient for each pixel of the input image.
def image_gradients_polar(image, sigma):
    image_dx, image_dy = gauss_derivs(image, sigma)
    magnitude = np.sqrt(image_dx**2 + image_dy**2)
    direction = np.arctan2(image_dx, image_dy)
    return magnitude, direction


# Laplace returns an image with the Laplacian-of-Gaussian of each pixel if the image
def laplace(image, sigma):
    image_dxx, image_dxy, image_dyy = gauss_second_derivs(image, sigma)
    return image_dxx + image_dyy


def get_edges(image, sigma, theta):
    grad_mag, grad_dir = image_gradients_polar(image, sigma)
    height, width = image.shape

    for i in range(height):
        for j in range(width):
            if grad_mag[i, j] > theta:
                pass
            else:
                grad_mag[i, j] = 0
    return grad_mag        
    


# What difficulties do you observe when selecting sigma and theta?
# Ans: as theta increases, edges with smaller gradient magnitude start to vanish
# leading to less distinct edges

def nms_for_canny(grad_mag, grad_dir):
    result = np.zeros_like(grad_mag)

    # Pre-define pixel index offset along different orientation
    offsets_x = [-1, -1, 0, 1, 1, 1, 0, -1, -1]
    offsets_y = [0, -1, -1, -1, 0, 1, 1, 1, 0]
    height, width = grad_mag.shape
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            pass
    return result

# def get_edges_with_nms(image, sigma, theta):
#     # YOUR CODE HERE
#     raise NotImplementedError()

# edges1 = get_edges(gray_im, sigma=2, theta=5)
# edges2 = get_edges_with_nms(
#     gray_im, sigma=2, theta=0.17
# )  # 0.17 corresponds to an absolute threshold of 5

# plot_multiple([edges1, edges2], ["get_edges", "get_edges_with_nms"], imsize=6)



# # def my_canny(image, sigma, theta_low, theta_high):
# #     # Compute gradient, apply non-maximum suppression
# #     # YOUR CODE HERE

# #     # Compute absolute threshold relative to max value
# #     max_val = np.max(grad_mag)
# #     theta_low_abs = theta_low * max_val
# #     theta_high_abs = theta_high * max_val

# #     # Initialize flags
# #     # Declare all pixels below the low threshold as visited
# #     # so edges are not followed there
# #     visited = grad_mag < theta_low_abs

# #     # Mark boundary pixels as visited
# #     visited[:, 0] = 1
# #     visited[:, -1] = 1
# #     visited[0, :] = 1
# #     visited[-1, :] = 1

# #     # Output image
# #     image_out = np.zeros_like(image)

# #     def follow_edge(x, y):
# #         visited[y, x] = True
# #         image_out[y, x] = 255

# #         # Pre-define pixel index offset along different orientation
# #         offsets_x = [-1, -1, 0, 1, 1, 1, 0, -1]
# #         offsets_y = [0, -1, -1, -1, 0, 1, 1, 1]

# #         for ox, oy in zip(offsets_x, offsets_y):
# #             # Note: `visited` is already False for points
# #             # below the low threshold.

# #             # YOUR CODE HERE
# #             raise NotImplementedError()

# #     is_high = grad_mag >= theta_high_abs
# #     # Main loop
# #     for x in range(image.shape[1]):
# #         for y in range(image.shape[0]):
# #             # YOUR CODE HERE
# #             raise NotImplementedError()

# #     return image_out

# # edge_canny = my_canny(gray_im, sigma=2, theta_low=0.1, theta_high=0.3)

# # blurred_cv = cv2.GaussianBlur(gray_im, ksize=(7, 7), sigmaX=2)
# # edge_canny_cv = cv2.Canny(blurred_cv.astype(np.uint8), 39, 72, L2gradient=True).astype(
# #     np.float32
# # )

# # plot_multiple(
# #     [edge_canny, edge_canny_cv, edge_canny - edge_canny_cv],
# #     ["my_canny", "cv2.Canny", "Difference"],
# #     imsize=5,
# # )
