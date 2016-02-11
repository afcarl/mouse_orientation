from __future__ import division
import numpy.random as npr

from mouse_orientation.load import load_training_data
from mouse_orientation.nnet import init_gmlp
from mouse_orientation.optimize import adadelta, make_batches
from mouse_orientation.orientation_regression import make_regression
from mouse_orientation.util import flatten

if __name__ == "__main__":
    npr.seed(0)

    images, angles = load_training_data('data/labeled_images.pkl')

    imsize = images.shape[1]
    hdims = [20]
    L2_reg = 1.

    paramvec, unflatten = flatten(init_gmlp(hdims, imsize, 1))
    loss = make_regression(L2_reg, unflatten)

    def callback(vals):
        print vals

    paramvec = adadelta(paramvec, loss, make_batches(100, images, angles), callback=callback)
