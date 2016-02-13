from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import operator as op
import cPickle as pickle
import os
from skimage.measure import label, regionprops
from scipy.ndimage.interpolation import rotate

each_side = 3
ellipse_image_threshold = 15.
resultsfile = 'data/labeled_images.pkl'

labeled = lambda: None
def press(event):
    key = event.key
    idx = perm[i]

    if key in ('U', 'D', 'A'):
        unrotated_panel = [data[idx]]
        key = key.lower()
    else:
        unrotated_panel = data[idx-each_side:idx+each_side+1]

    if key in ('u', 'd', 'a'):
        for im in unrotated_panel:
            labeled.images.append((im, get_angle(im), key))
        print 'labeled {} images'.format(len(unrotated_panel))
        if npr.rand() < 0.1:
            save()
            print "you've labeled {} images, wow!".format(len(labeled.images))

    if key != 'shift':
        next_slide()

def next_slide():
    global i
    i = i + 1
    draw_current_image()

def get_current_panel():
    idx = perm[i]
    images = map(lambda im: rot(im, get_angle(im)), data[idx-each_side:idx+each_side+1])
    return images

def draw_current_image():
    ax.matshow(np.hstack(get_current_panel()), cmap='plasma')
    plt.draw()

def get_angle(im):
    props = sorted(regionprops(label(im > ellipse_image_threshold)), key=op.attrgetter('area'))[-1]
    return np.pi/2. - props.orientation

def rot(im, angle):
    return rotate(im, np.rad2deg(angle), reshape=False)

def save():
    with open(resultsfile, 'w') as outfile:
        pickle.dump((labeled.images, i), outfile, protocol=-1)

def show_results(sidelen=5):
    random_subset = lambda lst, num: [lst[i] for i in npr.permutation(len(lst))[:num]]
    align = lambda im, angle, label: rot(im, (angle if label == 'u' else np.pi + angle))
    images, angles, labels = zip(*random_subset(labeled.images, sidelen**2))
    ims = iter(map(align, images, angles, labels))
    bigmat = np.vstack([np.hstack([ims.next() for j in range(sidelen)]) for i in range(sidelen)])
    plt.matshow(bigmat, cmap='plasma')

if __name__ == "__main__":
    plt.ion()

    if os.path.isfile(resultsfile):
        with open(resultsfile, 'r') as infile:
            labeled.images, i = pickle.load(infile)
    else:
        labeled.images = []
        i = 0

    # globals
    data = np.load('data/sod1-images-norotation.npy')
    perm = npr.RandomState(seed=0).permutation(data.shape[0])
    fig, ax = plt.subplots(figsize=(12,3))

    fig.canvas.mpl_connect('key_press_event', press)
    draw_current_image()
