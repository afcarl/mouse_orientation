from __future__ import division
import numpy as np
import numpy.random as npr
from skimage.measure import label, regionprops
from scipy.ndimage.morphology import binary_opening, binary_dilation
from util import rotate
import cPickle as pickle
import operator as op

wrap_angle = lambda angle: angle % (2*np.pi)

def clean_frames(frames, return_mask=False):
    mask = binary_dilation(binary_opening(frames > 10., iterations=2), iterations=1)
    out = frames.copy()
    out[~mask] = 0.
    return out if not return_mask else (out, mask)

def load_training_data(filename, augmentation=0, filter_out_ambiguous=True):
    print 'loading training data...'

    with open(filename, 'r') as infile:
        train_tuples, _ = pickle.load(infile)

    if filter_out_ambiguous:
        not_ambiguous = lambda tup: tup[-1] != 'a'
        train_tuples = filter(not_ambiguous, train_tuples)

    def flip(angle, label):
        flipme = label == 'd' or (label == 'a' and npr.uniform() < 1./2)
        return wrap_angle(np.pi + angle) if flipme else angle

    images, partial_angles, labels = zip(*train_tuples)
    angles = np.array(map(flip, partial_angles, labels))
    images = clean_frames(np.array(images))

    if augmentation > 0:
        random_rotations = lambda size: npr.uniform(0, 2*np.pi, size=size)

        def augment(images, angles):
            def rot(pair, rotation):
                im, angle = pair
                return rotate(im, rotation), wrap_angle(angle - rotation)

            return zip(*map(rot, zip(images, angles), random_rotations(len(angles))))

        aug_images, aug_angles = zip(*[augment(images, angles) for _ in xrange(augmentation)])
        images = np.concatenate((images,) + aug_images)
        angles = np.concatenate((angles,) + aug_angles)

    perm = npr.permutation(images.shape[0])
    images, angles = images[perm], angles[perm]

    assert np.all(angles >= 0.) and np.all(angles <= 2*np.pi)

    print '...done loading {} frames ({} raw, augmented to {}x)'.format(
        images.shape[0], len(train_tuples), 1+augmentation)
    return images.reshape(images.shape[0], -1), angles
