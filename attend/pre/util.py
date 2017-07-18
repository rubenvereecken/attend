import os
import numpy as np

def find_deepest_dirs(path='.'):
    for dir, subdirs, files in os.walk(path):
        if len(subdirs) == 0:
            yield dir



from itertools import zip_longest

def batch_pad(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def batch(iterable, n):
    for batch in batch_pad(iterable, n, None):
        yield list(filter(lambda x: not x is None, batch))


def makedirs_if_needed(file_or_dir):
    if '/' in file_or_dir:
        path = '/'.join(file_or_dir.split('/')[:-1])
        os.makedirs(path, exist_ok=True)


def rm_if_needed(file):
    if os.path.exists(file):
        os.remove(file)


# https://github.com/albanie/pts_loader
def load_pts(path):
    """takes as input the path to a .pts and returns a list of
	tuples of floats containing the points in in the form:
	[(x_0, y_0, z_0),
	 (x_1, y_1, z_1),
	 ...
	 (x_n, y_n, z_n)]"""
    with open(path) as f:
        rows = [rows.strip() for rows in f]

    """Use the curly braces to find the start and end of the point data"""
    head = rows.index('{') + 1
    tail = rows.index('}')

    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    """Convert entries from lists of strings to tuples of floats"""
    points = [tuple([float(point) for point in coords]) for coords in coords_set]
    return np.array(points)


class LengthyGenerator(dict):
    """
    Generator wrapper for when the generator length is known beforehand.
    Plays nicely with tqdm, for example.
    Also supports a minimal `dict` interface for attaching metadata.
    """

    def __init__(self, generator, length):
        self.generator = generator
        self.length = length
        self._dict = {}

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)



def float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def float32_featurelist(value):
    return tf.train.FeatureList(feature=list(map(_float32_feature, value)))
def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
