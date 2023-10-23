import numpy as np
import cv2
import imageio.v2 as iio
import matplotlib.pyplot as plt

im = iio.imread("flamingo-1024x1550.webp")


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
    
    
def prewitt_filter(im):
    prewitt_filt_x = np.array([[-1, 0 , 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_filt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    height, width, layers = im.shape
    padded_im = np.pad(im, pad_width=((3, 3), (3, 3), (0, 0)), constant_values=0, mode="constant")
    temp_im = np.zeros_like(padded_im)
    final_im = np.zeros_like(padded_im)
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                temp_im[i, j, k] = np.sum(padded_im[i:i+3, j:j+3, k] * prewitt_filt_x)
    
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                final_im[i, j, k] = np.sum(temp_im[i:i+3, j:j+3, k] * prewitt_filt_y)
    
    return final_im[:, :, 0]

def sorbet_filter(im):
    sorbet_filt_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sorbet_filt_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    height, width, layers = im.shape
    padded_im = np.pad(im, pad_width=((3, 3), (3, 3), (0, 0)), constant_values=0, mode="constant")
    temp_im = np.zeros_like(padded_im)
    final_im = np.zeros_like(padded_im)
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                temp_im[i, j, k] = np.sum(padded_im[i:i+3, j:j+3, k] * sorbet_filt_x)
    
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                final_im[i, j, k] = np.sum(temp_im[i:i+3, j:j+3, k] * sorbet_filt_y)
    
    return final_im[:, :, 0]

def robert_filter(im):
    robert_filt_x = np.array([[0, 1], [-1, 0]])
    robert_filt_y = np.array([[1, 0], [0, -1]])
    height, width, layers = im.shape
    padded_im = np.pad(im, pad_width=((2, 2), (2, 2), (0, 0)), constant_values=0, mode="constant")
    temp_im = np.zeros_like(padded_im)
    final_im = np.zeros_like(padded_im)
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                temp_im[i, j, k] = np.sum(padded_im[i:i+2, j:j+2, k] * robert_filt_x)
    
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                final_im[i, j, k] = np.sum(temp_im[i:i+2, j:j+2, k] * robert_filt_y)
    
    return final_im[:, :, 0]

    
def gamma_compression(im):
    height, width, layers = im.shape
    compressed_im = np.copy(im)
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                compressed_im[i, j, k] = np.sqrt(compressed_im[i, j, k])
    return compressed_im
    
def compute_gradients(im):
    filt = np.array([[-1, 0, 1]])
    padded_im = np.pad(comp_im, pad_width=((3, 3), (3, 3), (0, 0)), constant_values=0, mode="constant")
    height, width, layers = im.shape
    temp_im = np.zeros_like(padded_im)
    grad_im = np.zeros_like(padded_im)
    
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                temp_im[i, j, k] = np.sum(padded_im[i, j:j+3, k] * filt)
                
    for i in range(height):
        for j in range(width):
            for k in range(layers - 1):
                grad_im[i, j, k] = np.sum(temp_im[i:i+3, j, k] * filt)
                
    return grad_im[:, :, 0]
                
comp_im = gamma_compression(im)

plot_multiple([im, comp_im, compute_gradients(comp_im), prewitt_filter(comp_im), sorbet_filter(comp_im), robert_filter(comp_im)],
              ["Original", "Gamma Compressed", "Gradient", "Prewitt", "Sorbet", "Robert"])
