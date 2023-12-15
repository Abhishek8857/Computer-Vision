from convinience_functions import load_image, draw_keypoints
from nms import nms
from hessian_detector import hessian_scores, score_map_to_keypoints
from plot import plot_multiple
import matplotlib.pyplot as plt


class HessianOpts:
    sigma1: float = 3
    score_threshold: float = 5e-4


opts = HessianOpts()

paths = ["images/checkboard.jpg", "images/graf.png", "images/gantrycrane.png"]

images = []
titles = []
for path in paths:
    image = load_image(path)
    score_map = hessian_scores(image, opts)
    score_map_nms = nms(score_map)
    keypoints = score_map_to_keypoints(score_map_nms, opts)
    keypoint_image = draw_keypoints(image, keypoints)
    images += [score_map, keypoint_image]
    titles += ["Hessian scores", "Hessian keypoints"]

plot_multiple(images, titles, max_columns=2, colormap="viridis")
plt.show()