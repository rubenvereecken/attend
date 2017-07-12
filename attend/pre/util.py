import os

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
