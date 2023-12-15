import imageio.v3 as iio
import numpy as np
import scipy.ndimage as ndimage
import cv2



# Many useful functions
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