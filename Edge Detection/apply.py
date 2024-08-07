import imageio.v3 as iio
from convinience_functions import *
from edge_detection import *
import cv2

# Read image
color_im = iio.imread("Computer-Vision/images/Marq_3.jpg")
# Convert to grayscale
gray_im = cv2.cvtColor(color_im, cv2.COLOR_RGB2GRAY).astype(np.float32)

plot_multiple([get_edges(gray_im, 2, 2)], ["Image"])
              
sigmas = [1, 2]
thetas = [1, 2, 5, 10]

images = []
titles = []

for sigma in sigmas:
    for theta in thetas:
        edges = get_edges(gray_im, sigma, theta)
        images.append(edges)
        titles.append(f"sigma={sigma}, theta={theta}")

plot_multiple(images, titles, max_columns=4, imsize=2)


mag, dir = image_gradients_polar(gray_im, 2)
nms_for_canny(grad_mag=mag, grad_dir=dir)
