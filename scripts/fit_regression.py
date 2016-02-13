from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from functools import partial

from mouse_orientation.load import load_training_data
from mouse_orientation.nnet import init_gmlp
from mouse_orientation.regression import make_regression, empirical_l2_reg
from mouse_orientation.optimize import adadelta, rmsprop, adam
from mouse_orientation.viz import plot_images_and_angles
from mouse_orientation.util import flatten

save_figs = True


def update_training_progress(fig, ax, line, vals):
    line.set_data(range(1, len(vals)+1), vals)
    ax.set_ylim(min(vals), np.percentile(vals, 99))
    ax.set_xlim(1, len(vals)+1)
    fig.draw_artist(line)
    if save_figs: fig.savefig('training.png')
    plt.pause(1e-6)

if __name__ == "__main__":
    npr.seed(0)
    plt.ion()

    # load training data and plot some examples
    images, angles = load_training_data('data/labeled_images.pkl', augmentation=19)
    plot_images_and_angles(images[:20], angles[:20])

    N, imsize = images.shape
    hdims = [50, 50]
    # l2_reg = empirical_l2_reg(images, hdims)
    l2_reg = 0.

    paramvec, unflatten = flatten(init_gmlp(hdims, imsize, 1))
    predict, loss, prediction_error = make_regression(l2_reg, unflatten)

    # make a subset to show predictions on
    test_subset = npr.RandomState(0).choice(images.shape[0], size=20, replace=False)
    test_im, test_angle = images[test_subset], angles[test_subset]

    # make a figure for training progress
    fig, ax = plt.subplots()
    line, = ax.plot([])
    plt.draw()

    # make a figure for showing predictions
    prediction_fig = plot_images_and_angles(test_im, predict(test_im, paramvec))

    def callback(epoch, paramvec, vals):
        print 'epoch {}: {}'.format(epoch, prediction_error(paramvec, test_im, test_angle))
        update_training_progress(fig, ax, line, vals)
        plot_images_and_angles(test_im, predict(test_im, paramvec), prediction_fig)
        if save_figs: prediction_fig.figure.savefig('orientations.png')

    # optimize
    data = (images, angles)
    paramvec = adam(data, paramvec, loss,
                    batch_size=100, rate=1e-3, epochs=25, callback=callback)
    paramvec = adam(data, paramvec, loss,
                    batch_size=250, rate=1e-3, epochs=50, callback=callback)
    paramvec = adam(data, paramvec, loss,
                    batch_size=1000, rate=5e-4, epochs=300, callback=callback)
