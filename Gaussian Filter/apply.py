import imageio.v3 as iio
from gaussian_filter import *
from plot import plot_multiple

# Read the image
im = iio.imread("images\Quart_1.jpg")

# Print the Guassian Value
print("Gauss of 5 is", gauss(5, sigma=2))

# Plot the blurred images
sigmas = [2, 4, 8]
blurred_images = [gaussian_filter(im, s) for s in sigmas]
titles_classic = [f"Classic: sigma={s}" for s in sigmas]
plot_multiple(blurred_images, titles_classic)

# 
blurred_images_cv = [gauss_cv(im, s) for s in sigmas]
differences = [abs_diff(x, y) for x, y in zip(blurred_images, blurred_images_cv)]
titles_cv = [f"cv2: sigma={s}" for s in sigmas]
titles_diff = [f"Classic - cv2: sigma={s}" for s in sigmas]

# Plot to visualise the difference
plot_multiple(blurred_images_cv, titles_cv)
plot_multiple(differences, titles_diff)

plt.show()