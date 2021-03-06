from __future__ import division
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.core import getval
from scipy.ndimage.interpolation import rotate as _rotate

# from mindbike repo, by dougal
def flatten(value):
    # value can be any nested thing ((), array, [] ) etc
    # returns numpy array
    if isinstance(getval(value), np.ndarray):
        def unflatten(vector):
            return np.reshape(vector, value.shape)
        return np.ravel(value), unflatten

    elif isinstance(getval(value), float):
        return np.array([value]), lambda x : x[0]

    elif isinstance(getval(value), tuple):
        if not value:
            return np.array([]), lambda x : ()
        flattened_first, unflatten_first = flatten(value[0])
        flattened_rest, unflatten_rest = flatten(value[1:])
        def unflatten(vector):
            N = len(flattened_first)
            return (unflatten_first(vector[:N]),) + unflatten_rest(vector[N:])

        return np.concatenate((flattened_first, flattened_rest)), unflatten

    elif isinstance(getval(value), list):
        if not value:
            return np.array([]), lambda x : []

        flattened_first, unflatten_first = flatten(value[0])
        flattened_rest, unflatten_rest = flatten(value[1:])
        def unflatten(vector):
            N = len(flattened_first)
            return [unflatten_first(vector[:N])] + unflatten_rest(vector[N:])

        return np.concatenate((flattened_first, flattened_rest)), unflatten

    else:
        raise Exception("Don't know how to flatten type {}".format(type(value)))

# from mindbike repo, by matt and dougal
def curry(f, N=None):
    if N is None:
        N = len(getargspec(f).args)

    def curried_f(*args):
        num_unbound = N - len(args)
        if num_unbound == 0:
            return f(*args)
        else:
            return curry(partial(f, *args), N=num_unbound)

    return curried_f

def rotate(im, angle):
    return _rotate(im, np.rad2deg(angle), reshape=False)

# based on syllables utility function
def colorize(frames, cmap):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    frames = frames.copy()
    frames -= np.nanmin(frames)
    frames /= np.nanmax(frames)
    return np.clip(cmap(frames)[...,:3]*255, 0, 255)

# based on syllables utility function
def make_movie(frames, outfile, fps=30, cmap='cubehelix'):
    import moviepy.editor as mp
    frames = colorize(frames, cmap)
    duration = (frames.shape[0] - 1) / fps
    ani = mp.VideoClip(make_frame=lambda t: frames[int(t*fps)], duration=duration)
    ani.write_videofile(outfile, fps=fps)
