# distutils: extra_compile_args = -O2 -w
# cython: boundscheck = False, nonecheck = False, wraparound = False, cdivision = True

import numpy as np
cimport numpy as np

def kalman_filter(
    double mu_init, double sigmasq_init, double a, double sigmasq_states,
    double[::1] node_mu, double[::1] node_sigmasq):
    cdef int T = node_mu.shape[0]
    cdef int t

    cdef double predict_mu = mu_init
    cdef double predict_sigmasq = sigmasq_init

    cdef double[::1] filtered_mu = np.zeros(T)
    cdef double[::1] filtered_sigmasq = np.zeros(T)

    for t in range(T):
        # condition
        filtered_sigmasq[t] = 1./(1./predict_sigmasq + 1./node_sigmasq[t])
        filtered_mu[t] = filtered_sigmasq[t] * (
            predict_mu / predict_sigmasq + node_mu[t] / node_sigmasq[t])

        # predict
        predict_mu = a * filtered_mu[t]
        predict_sigmasq = a**2 * filtered_sigmasq[t] + sigmasq_states

    return np.asarray(filtered_mu), np.asarray(filtered_sigmasq)

def kalman_smoother(
    double mu_init, double sigmasq_init, double a, double sigmasq_states,
    double[::1] node_mu, double[::1] node_sigmasq):
    cdef int T = node_mu.shape[0]
    cdef int t

    # temps
    cdef double predict_mu = mu_init
    cdef double predict_sigmasq = sigmasq_init
    cdef double G

    # outputs
    cdef double[::1] filtered_mu = np.zeros(T)
    cdef double[::1] filtered_sigmasq = np.zeros(T)

    # run filter fowrard
    for t in range(T):
        # condition
        filtered_sigmasq[t] = 1./(1./predict_sigmasq + 1./node_sigmasq[t])
        filtered_mu[t] = filtered_sigmasq[t] * (
            predict_mu / predict_sigmasq + node_mu[t] / node_sigmasq[t])

        # predict
        predict_mu = a * filtered_mu[t]
        predict_sigmasq = a**2 * filtered_sigmasq[t] + sigmasq_states

    # run rts backward (see Thm. 8.2 of Sarkka 2013)
    for t in range(T-1, 0, -1):
        predict_mu = a * filtered_mu[t-1]
        predict_sigmasq = a**2 * filtered_sigmasq[t-1] + sigmasq_states
        G = a * filtered_sigmasq[t-1] / predict_sigmasq
        filtered_mu[t-1] += filtered_mu[t-1] + G * (filtered_mu[t] - predict_mu)
        filtered_sigmasq[t-1] += G**2 * (filtered_sigmasq[t] - predict_sigmasq)

    return np.asarray(filtered_mu), np.asarray(filtered_sigmasq)
