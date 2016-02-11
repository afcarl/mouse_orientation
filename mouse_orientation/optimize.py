from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad as vgrad


def make_batches(batch_size, *data):
    N = data[0].shape[0]
    slices = [slice(i, min(i+batch_size, N)) for i in range(0, N, batch_size)]
    return [[d[sl] for d in data] for sl in slices]

def adadelta(paramvec, loss, batches, epochs=1, rho=0.95, epsilon=1e-6, callback=None):
    sum_gsq = np.zeros_like(paramvec)
    sum_usq = np.zeros_like(paramvec)
    vals = []

    for epoch in range(epochs):
        permuted_batches = [batches[i] for i in npr.permutation(len(batches))]
        for im, angle in permuted_batches:
            val, g = vgrad(loss)(paramvec, im, angle)
            sum_gsq = rho*sum_gsq + (1.-rho)*g**2
            ud = -np.sqrt(sum_usq + epsilon) / np.sqrt(sum_gsq + epsilon) * g
            sum_usq = rho*sum_usq + (1.-rho)*ud**2
            paramvec = paramvec + ud
            vals.append(val)
        if callback: callback(paramvec, vals, permuted_batches)
    return paramvec

def rmsprop(paramvec, loss, batches, rate, epochs=1, rho=0.9, epsilon=1e-6, callback=None):
    sumsq = np.zeros_like(paramvec)
    vals = []

    for epoch in range(epochs):
        permuted_batches = [batches[i] for i in npr.permutation(len(batches))]
        for im, angle in permuted_batches:
            val, g = vgrad(loss)(paramvec, im, angle)
            sumsq = rho*sumsq + (1.-rho)*g**2
            paramvec = paramvec - rate * g / np.sqrt(sumsq + epsilon)
            vals.append(val)
        if callback: callback(paramvec, vals, permuted_batches)
    return paramvec

def adam(paramvec, loss, batches, rate, epochs=1, b1=0.9, b2=0.999, epsilon=1e-8, callback=None):
    m = np.zeros_like(paramvec)
    v = np.zeros_like(paramvec)
    vals = []

    for epoch in range(epochs):
        permuted_batches = [batches[i] for i in npr.permutation(len(batches))]
        for im, angle in permuted_batches:
            val, g = vgrad(loss)(paramvec, im, angle)
            m = (1. - b1)*g    + b1*m
            v = (1. - b2)*g**2 + b2*v
            mhat = m / (1 - b1**(i+1))
            vhat = v / (1 - b2**(i+1))
            paramvec -= rate * mhat / (np.sqrt(vhat) + epsilon)
            vals.append(val)
        if callback: callback(paramvec, vals, permuted_batches)
    return paramvec