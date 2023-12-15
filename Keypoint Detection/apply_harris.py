from dataclasses import dataclass, field
from typing import Callable

from convinience_functions import load_image, draw_keypoints
from nms import nms
from harris_detector import harris_scores, score_map_to_keypoints
from plot import plot_multiple
import matplotlib.pyplot as plt


class HarrisOpts:
    sigma1: float = 2
    sigma2: float = 2 * 2
    alpha: float = 0.06
    score_threshold: float = 1e-8


opts = HarrisOpts()


paths = ["images/checkboard.jpg", "images/graf.png", "images/gantrycrane.png"]
images = []
titles = []
for path in paths:
    image = load_image(path)

    score_map = harris_scores(image, opts)
    score_map_nms = nms(score_map)
    keypoints = score_map_to_keypoints(score_map_nms, opts)
    keypoint_image = draw_keypoints(image, keypoints)

    images += [score_map, keypoint_image]
    titles += ["Harris response scores", "Harris keypoints"]
plot_multiple(images, titles, max_columns=2, colormap="viridis")
plt.show()

