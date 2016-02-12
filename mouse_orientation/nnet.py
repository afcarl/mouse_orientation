from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def compose(funcs):
    return reduce(lambda f, g: lambda x: g(f(x)), funcs, lambda x: x)

def make_layer(activation):
    def layer(W, b):
        def apply(h):
            return activation(np.dot(h, W) + b)
        return apply
    return layer

tanh_layer = make_layer(np.tanh)
sigmoid_layer = make_layer(sigmoid)
linear_layer = make_layer(lambda x: x)

def init_layer(shape, scale=1e-2):
    m, n = shape
    return scale*npr.randn(m, n), scale*npr.randn(n)

def gmlp(x, params):
    nnet_params, ((W_mu, b_mu), (W_sigma, b_sigma)) = params[:-2], params[-2:]

    nnet = compose(tanh_layer(W, b) for W, b in nnet_params)
    mu = linear_layer(W_mu, b_mu)
    log_sigmasq = linear_layer(W_sigma, b_sigma)

    nnet_outputs = nnet(x)
    return mu(nnet_outputs), log_sigmasq(nnet_outputs)

def init_gmlp(hdims, n, p, scale=1e-2):
    dims = [n] + hdims
    nnet_params = map(lambda shape: init_layer(shape, scale), zip(dims[:-1], dims[1:]))
    W_mu, b_mu = init_layer((dims[-1], p), scale)
    W_sigma, b_sigma = init_layer((dims[-1], p), scale)
    return nnet_params + [(W_mu, b_mu), (W_sigma, b_sigma)]
