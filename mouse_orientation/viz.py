from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from util import rotate


def plot_images_and_angles(images, angles):
    def plot_angle(center, angle):
        x0, y0 = center
        x1, y1 = x0 - 10*np.sin(-angle), y0 - 10*np.cos(-angle)
        plt.plot([x0, x1], [y0, y1], '-', color='cyan', linewidth=2)

    make_pair = lambda im, angle: np.vstack((im, rotate(im, angle)))
    plt.matshow(np.hstack(map(make_pair, images.reshape(-1, 30, 30), angles)), cmap='plasma')

    plt.autoscale(False)
    # plt.axis('off')
    xs = -0.5 + 15 + 30*np.arange(images.shape[0])
    ys = np.repeat(15, images.shape[0])
    centers = zip(xs, ys)
    for center, angle in zip(centers, angles):
        plot_angle(center, angle)
