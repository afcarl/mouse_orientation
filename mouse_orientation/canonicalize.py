from __future__ import division
import numpy as np

from load import clean_frames
from orientation_regression import predict
from kalman import kalman_smoother
from util import rotate


def infer_angles(nnet_params, frames, sigmasq_states):
    frames = clean_frames(frames).reshape(frames.shape[0], -1)
    mus, log_sigmasqs = predict(frames, nnet_params)
    angles, _ = kalman_smoother(
        0., 100., 1., sigmasq_states, np.unwrap(mus), np.exp(log_sigmasqs))
    return angles

def reorient_video(nnet_params, frames, sigmasq_states):
    if frames.ndim != 3: raise ValueError
    angles = infer_angles(nnet_params, frames, sigmasq_states)
    return np.array(map(rotate, frames, angles))
