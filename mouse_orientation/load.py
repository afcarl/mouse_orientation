from __future__ import division
import numpy as np
import numpy.random as npr
from scipy.ndimage.interpolation import rotate
import cPickle as pickle

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
