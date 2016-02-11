from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from util import rotate


def plot_images_and_angles(images, angles, im=None):
    def plot_angle(ax, center, angle, line=None):
        x0, y0 = center
        x1, y1 = x0 - 10*np.sin(-angle), y0 - 10*np.cos(-angle)
        if not line:
            ax.plot([x0, x1], [y0, y1], '-', color='cyan', linewidth=2)
        else:
            line.set_data([x0, x1], [y0, y1])

    make_pair = lambda im, angle: np.vstack((im, rotate(im, angle)))

    bigmat = np.hstack(map(make_pair, images.reshape(-1, 30, 30), angles))
    if im is None:
        im = plt.matshow(bigmat, cmap='plasma')
    else:
        im.set_data(bigmat)
    ax = im.axes

    ax.autoscale(False)
    ax.axis('off')
    xs = -0.5 + 15 + 30*np.arange(images.shape[0])
    ys = np.repeat(15, images.shape[0])
    centers = zip(xs, ys)

    if not ax.lines:
        for center, angle in zip(centers, angles):
            plot_angle(ax, center, angle)
    else:
        for center, angle, line in zip(centers, angles, ax.lines):
            plot_angle(ax, center, angle, line)

    plt.draw()
    return im
