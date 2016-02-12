from __future__ import division
import autograd.numpy as np
from autograd.scipy.misc import logsumexp

from nnet import gmlp, init_gmlp, sigmoid
from util import flatten

def gaussian_loglike(x, mu, log_sigmasq):
    return np.mean(logsumexp(
        -0.5*((np.log(2*np.pi) + log_sigmasq) + (x - mu)**2. / np.exp(log_sigmasq)),
        axis=0))

def predict(im, params):
    mu, log_sigmasq = map(np.squeeze, gmlp(im, params))
    return 2*np.pi*sigmoid(mu), 4.*np.tanh(log_sigmasq / 4.)

def empirical_l2_reg(images, hdims):
    l2 = init_gmlp(hdims, images.shape[1], 1, scale=0.)
    W_1, b_1 = l2[0]
    W_1[:] = 1. / (0.01 + np.var(images, axis=0)[:,None])
    return flatten(l2)[0]

def make_regression(L2_reg, unflatten):
    def flat_predict(im, paramvec):
        return predict(im, unflatten(paramvec))

    def loglike(theta, prediction):
        theta_hat, log_sigmasq_hat = prediction
        effective_thetas = theta + np.array([-2*np.pi, 0., 2*np.pi])[:,None]
        return gaussian_loglike(effective_thetas, theta_hat, log_sigmasq_hat)

    def logprior(paramvec):
        return -1./2 * np.dot(paramvec, L2_reg * paramvec)

    def loss(paramvec, im, angle):
        return - logprior(paramvec) - loglike(angle, flat_predict(im, paramvec))

    def prediction_error(paramvec, im, angle):
        predicted_angle, _ = flat_predict(im, paramvec)
        return np.sqrt(np.mean((predicted_angle - angle)**2))

    return flat_predict, loss, prediction_error
