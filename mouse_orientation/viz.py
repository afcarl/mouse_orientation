from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from util import rotate


def plot_images_and_angles(images, prediction, im=None):
    if isinstance(prediction, tuple):
        angles, log_sigmasqs = prediction
    else:
        angles, log_sigmasqs = prediction, np.zeros_like(prediction)

    def plot_angle(ax, center, angle, sigmasq, line=None):
        precision = 1./sigmasq
        x0, y0 = center
        x1, y1 = x0 - precision*np.sin(-angle), y0 - precision*np.cos(-angle)
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
        for center, angle, log_sigmasq in zip(centers, angles, log_sigmasqs):
            plot_angle(ax, center, angle, np.exp(log_sigmasq))
    else:
        for center, angle, log_sigmasq, line in zip(centers, angles, log_sigmasqs, ax.lines):
            plot_angle(ax, center, angle, np.exp(log_sigmasq), line)

    plt.draw()
    return im
