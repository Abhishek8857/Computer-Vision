USE_GPU = True  # Set to True if you have installed tensorflow for GPU

import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_multiple(
    images,
    titles=None,
    colormap="gray",
    max_columns=np.inf,
    imwidth=2,
    imheight=2,
    share_axes=False,
):
    """
    Plot multiple images as subplots on a grid. Images must be channel-first
    and between [0, 1].
    """
    images = [np.transpose(im, (1, 2, 0)) for im in images]
    if titles is None:
        titles = [""] * len(images)
    assert len(images) == len(titles)
    n_images = len(images)
    n_cols = min(max_columns, n_images)
    n_rows = int(np.ceil(n_images / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * imwidth, n_rows * imheight),
        squeeze=False,
        sharex=share_axes,
        sharey=share_axes,
    )

    axes = axes.flat
    # Hide subplots without content
    for ax in axes[n_images:]:
        ax.axis("off")

    if not isinstance(colormap, (list, tuple)):
        colormaps = [colormap] * n_images
    else:
        colormaps = colormap

    for ax, image, title, cmap in zip(axes, images, titles, colormaps):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.tight_layout()


def visualize_dataset(dataset, labels, n_samples=24, max_columns=6):
    xs, ys = list(zip(*[dataset[i] for i in range(n_samples)]))
    plot_multiple(
        [x / 2 + 0.5 for x in xs], [labels[i] for i in ys], max_columns=max_columns)
    
# Code to download the CIFAR Dataset - Needs to be run only once

normalize_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

train_data = CIFAR10(
    root="cifar10/test/", train=True, download=False, transform=normalize_transform)
test_data = CIFAR10(
    root="cifar10/train/", train=False, download=False, transform=normalize_transform)

labels = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

augment_transform = transforms.Compose([
                            transforms.RandomApply(
                                torch.nn.ModuleList([
                                    transforms.RandomAffine(degrees=0, 
                                                            translate=(0.1, 0.1),
                                                            scale=(0.9, 1.1),
                                                            ), 
                                     transforms.RandomHorizontalFlip(p=1)
                                                    ]), p=0.5)])
    
augmented_train_data = CIFAR10(
    root="cifar10/train/",
    train=True,
    download=False,
    transform=transforms.Compose([augment_transform, normalize_transform]),
)


visualize_dataset(augmented_train_data, labels)


tanh_mlp = nn.Sequential(
nn.Flatten(),
nn.Linear(in_features=3072, out_features=512),
nn.Tanh(),
nn.Linear(in_features=512, out_features=10),    
)   
if USE_GPU:
    tanh_mlp.cuda()