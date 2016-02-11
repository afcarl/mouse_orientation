from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from functools import partial

from mouse_orientation.load import load_training_data
from mouse_orientation.nnet import init_gmlp
from mouse_orientation.orientation_regression import make_regression
from mouse_orientation.optimize import adadelta, rmsprop, adam, make_batches
from mouse_orientation.viz import plot_images_and_angles
from mouse_orientation.util import flatten


def update_training_progress(fig, ax, line, vals):
    line.set_data(range(1, len(vals)+1), vals)
    ax.set_ylim(min(vals), np.percentile(vals, 99))
    ax.set_xlim(1, len(vals)+1)
    fig.draw_artist(line)
    plt.pause(1e-6)

if __name__ == "__main__":
    npr.seed(0)
    plt.ion()

    # load training data and plot some examples
    images, angles = load_training_data('data/labeled_images.pkl', augmentation=5)
    plot_images_and_angles(images[:20], angles[:20])

    imsize = images.shape[1]
    hdims = [50, 50]
    L2_reg = 0.

    paramvec, unflatten = flatten(init_gmlp(hdims, imsize, 1))
    predict, loss, prediction_error = make_regression(L2_reg, unflatten)

    # make a subset to show predictions on
    test_subset = npr.RandomState(0).choice(images.shape[0], size=10, replace=False)
    test_im, test_angle = images[test_subset], angles[test_subset]

    # make a figure for training progress
    fig, ax = plt.subplots()
    line, = ax.plot([])

    # make a figure for showing predictions
    prediction_fig = plot_images_and_angles(test_im, predict(test_im, paramvec)[0])

    def callback(paramvec, vals, batches):
        print prediction_error(paramvec, test_im, test_angle)
        update_training_progress(fig, ax, line, vals)
        plot_images_and_angles(test_im, predict(test_im, paramvec)[0], prediction_fig)

    # optimize
    paramvec = adam(paramvec, prediction_error, make_batches(1000, images, angles),
                    rate=1e-3, epochs=250, callback=callback)
    paramvec = adam(paramvec, prediction_error, make_batches(2000, images, angles),
                    rate=5e-4, epochs=1000, callback=callback)
    # paramvec = adam(paramvec, prediction_error, [(images, angles)],
    #                 rate=1e-4, epochs=250, callback=callback)
