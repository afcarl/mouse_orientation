from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

from mouse_orientation.load import clean_frames
from mouse_orientation.canonicalize import infer_angles, reorient_video
from mouse_orientation.regression import predict
from mouse_orientation.kalman import kalman_smoother


if __name__ == "__main__":
    sigmasq_states = 0.1**2

    with open('data/nnet_params.pkl', 'r') as infile:
        nnet_params = pickle.load(infile)

    with open('data/test_frames_dict.pkl', 'r') as infile:
        datadict = pickle.load(infile)

    ### manual viz
    frames = clean_frames(datadict['test'])
    mus, log_sigmasqs = predict(frames.reshape(frames.shape[0], -1), nnet_params)
    plt.errorbar(np.arange(len(frames)), mus, np.exp(log_sigmasqs / 2.))
    plt.plot(kalman_smoother(
        0., 100., 1., sigmasq_states, mus, np.exp(log_sigmasqs))[0], 'g-')

    ### a look at results
    sl = slice(90, 110)
    # sl = slice(None)
    rotated_frames = reorient_video(nnet_params, frames, sigmasq_states)
    plt.matshow(np.vstack((np.hstack(frames[sl]), np.hstack(rotated_frames[sl]))), cmap='plasma')

    plt.show()
