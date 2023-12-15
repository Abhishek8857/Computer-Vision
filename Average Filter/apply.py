from average_filter import average_filter
from plot import plot_multiple
import imageio.v2  as iio

im = iio.imread("images\spnoise.png")


plot_multiple(images=(im, average_filter(im)),
              titles=("Original", "Filtered"))
