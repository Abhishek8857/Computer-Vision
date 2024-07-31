from median_filter import median_filter
import imageio.v3 as iio
from plot import plot_multiple
import matplotlib.pyplot as plt


im = iio.imread("Computer-Vision/images/spnoise.png")
filtered_im = median_filter(im)

plot_multiple(images=(im, filtered_im),
              titles=("Original", "Filtered"))
