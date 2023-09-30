import numpy as np
import cv2
import imageio.v2 as iio
import matplotlib.pyplot as plt

im = iio.imread("spnoise.png")


def plot_multiple(images, titles, colormap='gray', max_columns=np.inf, share_axes=True):
    """Plot multiple images as subplots on a grid."""
    assert len(images) == len(titles)
    n_images = len(images)
    n_cols = min(max_columns, n_images)
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4),
        squeeze=False, sharex=share_axes, sharey=share_axes)

    axes = axes.flat
    # Hide subplots without content
    for ax in axes[n_images:]:
        ax.axis('off')
        
    if not isinstance(colormap, (list,tuple)):
        colormaps = [colormap]*n_images
    else:
        colormaps = colormap

    for ax, image, title, cmap in zip(axes, images, titles, colormaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        
    fig.tight_layout()
    plt.show()
    
    
def median_filter_1(image):
    height, width, layers = image.shape
    image = np.pad(image, pad_width=((1, 1), (1, 1), (0, 0)), 
                   constant_values=0, 
                   mode="constant")
    for i in range(height):
        for j in range(width):
            for k in range(layers-1):
                temp = []
                for l in range(-1, 2):
                    for m in range(-1, 2):
                        temp.append(image[i+l, j+m, k])
                image[i, j, k] = np.divide(sum(temp), len(temp)).astype(int)
    return image

def median_filter_2(image):
    height, width = image.shape
    image = np.pad(image, pad_width=((1, 1), (1, 1)), 
                   constant_values=0, 
                   mode="constant")
    for i in range(height):
        for j in range(width):
            temp = []
            for l in range(-1, 2):
                for m in range(-1, 2):
                    temp.append(image[i+l, j+m])
            image[i, j] = np.divide(sum(temp), len(temp)).astype(int)
    return image

if len(im.shape) == 3:
    f_im = median_filter_1(im)
else:
    f_im = median_filter_2(im)
    
plot_multiple(images=(im, f_im),
              titles=("Original", "Filtered"))

