from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from util import rotate


def plot_images_and_angles(images, prediction, im=None, perrow=20):
    N = len(images)

    if isinstance(prediction, tuple):
        angles, log_sigmasqs = prediction
    else:
        angles, log_sigmasqs = prediction, np.zeros_like(prediction) - 1.

    def plot_angle(ax, center, angle, sigmasq, line=None):
        precision = 1./sigmasq
        x0, y0 = center
        x1, y1 = x0 - precision*np.sin(-angle), y0 - precision*np.cos(-angle)
        if not line:
            ax.plot([x0, x1], [y0, y1], '-', color='cyan', linewidth=2)
        else:
            line.set_data([x0, x1], [y0, y1])

    make_pair = lambda im, angle: np.vstack((im, rotate(im, angle)))
    pairs = map(make_pair, images.reshape(-1, 30, 30), angles)

    if N > perrow:
        nrows = N // perrow
        pairs= iter(pairs)
        bigmat = np.vstack([np.hstack([pairs.next() for j in range(perrow)]) for i in range(nrows)])
    else:
        bigmat = np.hstack(pairs)

    if im is None:
        im = plt.matshow(bigmat, cmap='plasma')
    else:
        im.set_data(bigmat)
    ax = im.axes

    ax.autoscale(False)
    ax.axis('off')

    if N > perrow:
        xs = np.tile(-0.5 + 15 + 30*np.arange(N // nrows), nrows)
        ys = np.repeat(-0.5 + 15 + 60*np.arange(nrows), N // nrows)
    else:
        xs = -0.5 + 15 + 30*np.arange(N)
        ys = np.repeat(15, N)
    centers = zip(xs, ys)

    if not ax.lines:
        for center, angle, log_sigmasq in zip(centers, angles, log_sigmasqs):
            plot_angle(ax, center, angle, np.exp(log_sigmasq))
    else:
        for center, angle, log_sigmasq, line in zip(centers, angles, log_sigmasqs, ax.lines):
            plot_angle(ax, center, angle, np.exp(log_sigmasq), line)

    plt.draw()
    return im
