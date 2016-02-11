from __future__ import division
import numpy as np

from orientation_regression import predict
from kalman import kalman_smoother
from util import rotate


def infer_angles(nnet_params, frames, sigmasq_states):
    frames = frames.reshape(frames.shape[0], -1)
    wrapped_mus, log_sigmasqs = predict(frames, nnet_params)
    mus, sigmasqs = np.unwrap(wrapped_mus), np.exp(log_sigmasqs)
    angles, _ = kalman_smoother(0., 100., 1., sigmasq_states, mus, sigmasqs)
    return angles

def reorient_video(nnet_params, frames, sigmasq_states):
    if frames.ndim != 3: raise ValueError
    angles = infer_angles(nnet_params, frames, sigmasq_states)
    return np.array(map(rotate, frames, angles))
