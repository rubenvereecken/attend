import os
import shutil
import numpy as np

def append_if(l, pred):
    def _append_if(name, v):
        if pred(name, v):
            l.append(v)
    return _append_if

def h5_find_all_groups(f):
    import h5py
    groups = []
    append_if_group = append_if(groups, lambda name, v: isinstance(v, h5py.Group))
    h5py.Group.visititems(f, append_if_group)
    return groups


def h5_find_deepest_groups(f):
    import h5py
    groups = []
    def _is_group_and_contains_datasets(name, v):
        return isinstance(v, h5py.Group) and len(v.keys()) > 0 and \
            any(map(lambda child: isinstance(child, h5py.Dataset), v.values()))
    maybe_append = append_if(groups, _is_group_and_contains_datasets)
    h5py.Group.visititems(f, maybe_append)
    return groups


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

def path_is_file(file_or_dir):
    return '.' in file_or_dir.split('/')[-1]


def makedirs_if_needed(file_or_dir):
    if path_is_file(file_or_dir):
        file_or_dir = '/'.join(file_or_dir.split('/')[:-1])
    if '/' in file_or_dir:
        path = file_or_dir
        print(path)
        os.makedirs(path, exist_ok=True)


def confirm(s):
    in_ = input(s)


def rm_if_needed(file, ask=False):
    if os.path.exists(file):
        if os.path.isdir(file) and ask:
            permission = input('{} is an existing directory, remove? (y/N)'.format(file))
            if permission.strip().lower() != 'y':
                return
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
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
