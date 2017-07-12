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
