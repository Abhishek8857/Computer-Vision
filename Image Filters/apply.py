import numpy as np
import cv2
import imageio.v2 as iio
from gamma_comp import gamma_compression
from plot import plot_multiple
from gradients import compute_gradients
from prewitt_filter import prewitt_filter
from robert_filter import robert_filter
from sorbet_filter import sorbet_filter


im = iio.imread("Computer-Vision/images/flamingo.webp")
comp_im = gamma_compression(im)

plot_multiple([im, comp_im, compute_gradients(comp_im), prewitt_filter(comp_im), sorbet_filter(comp_im), robert_filter(comp_im)],
              ["Original", "Gamma Compressed", "Gradient", "Prewitt", "Sobel", "Robert"])
