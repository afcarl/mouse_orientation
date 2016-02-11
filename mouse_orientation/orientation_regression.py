from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad as vgrad
from autograd.scipy.misc import logsumexp
import cPickle as pickle

from nnet import gmlp, init_gmlp, sigmoid
from util import flatten

def gaussian_loglike(x, mu, log_sigmasq):
    return np.mean(logsumexp(
        -0.5*((np.log(2*np.pi) + log_sigmasq) + (x - mu)**2. / np.exp(log_sigmasq)),
        axis=0))

def make_regression(L2_reg, unflatten):
    def predict(im, params):
        mu, log_sigmasq = gmlp(im, params)
        return np.squeeze(2*np.pi*sigmoid(mu)), np.squeeze(log_sigmasq)

    def loglike(theta, prediction):
        theta_hat, log_sigmasq_hat = prediction
        effective_thetas = theta + np.array([-2*np.pi, 0., 2*np.pi])[:,None]
        return gaussian_loglike(effective_thetas, theta_hat, log_sigmasq_hat)

    def logprior(paramvec):
        return -L2_reg * np.dot(paramvec, paramvec)

    def loss(paramvec, im, angle):
        return - logprior(paramvec) - loglike(angle, predict(im, unflatten(paramvec)))

    return loss

# TODO move this to load.py
from scipy.ndimage.interpolation import rotate

def load_training_data(filename, augmentation=0):
    def rot(pair, rotation):
        im, angle = pair
        return rotate(im, rotation, reshape=False), angle + rotation

    def augment(images, angles):
        return zip(*map(rot, zip(images, angles), npr.uniform(0, 2*np.pi, size=len(angles))))

    with open(filename, 'r') as infile:
        train_tuples, _ = pickle.load(infile)
    unpack = lambda (im, angle, label): (im, angle if label == 'u' else np.pi + angle)
    images, angles = map(np.array, zip(*map(unpack, train_tuples)))

    if augmentation > 0:
        aug_images, aug_angles = zip(*[augment(images, angles) for _ in xrange(augmentation)])
        images = np.concatenate((images,) + aug_images)
        angles = np.concatenate((angles,) + aug_angles)

    return images.reshape(images.shape[0], -1), angles

# TODO move these three functions to optimize.py
def make_batches(batch_size, *data):
    N = data[0].shape[0]
    slices = [slice(i, min(i+batch_size, N)) for i in range(0, N, batch_size)]
    return [[d[sl] for d in data] for sl in slices]

def adadelta(paramvec, loss, batches, epochs=1, rho=0.95, epsilon=1e-6, callback=None):
    sum_gsq = np.zeros_like(paramvec)
    sum_usq = np.zeros_like(paramvec)
    vals = []

    for epoch in range(epochs):
        for im, angle in npr.permutation(batches):
            val, g = vgrad(loss)(paramvec, im, angle)
            sum_gsq = rho*sum_gsq + (1.-rho)*g**2
            ud = -np.sqrt(sum_usq + epsilon) / np.sqrt(sum_gsq + epsilon) * g
            sum_usq = rho*sum_usq + (1.-rho)*ud**2
            paramvec = paramvec + ud
            vals.append(val)
        if callback: callback(vals)
    return paramvec

def rmsprop(paramvec, loss, batches, rate, epochs=1, rho=0.9, epsilon=1e-6, callback=None):
    sumsq = np.zeros_like(paramvec)
    vals = []

    for epoch in range(epochs):
        for im, angle in npr.permutation(batches):
            val, g = vgrad(loss)(paramvec, im, angle)
            sumsq = rho*sumsq + (1.-rho)*g**2
            paramvec = paramvec - rate * g / np.sqrt(sumsq + epsilon)
            vals.append(val)
        if callback: callback(vals)
    return paramvec


if __name__ == "__main__":
    npr.seed(0)

    images, angles = load_training_data('data/labeled_images.pkl')

    imsize = images.shape[1]
    hdims = [20]
    L2_reg = 1.

    paramvec, unflatten = flatten(init_gmlp(hdims, imsize, 1))
    loss = make_regression(L2_reg, unflatten)

    paramvec = adadelta(paramvec, loss, make_batches(100, images, angles))
