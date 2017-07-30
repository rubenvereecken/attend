import inspect
import numpy as np

def pick(dic, l):
    '''
    Pick from dictionary all the keys present in a list of keywords
    After the spirit of Lodash pick
    '''
    return { k: v for k, v in dic.items() if k in l }


def params_for(fun):
    return list(inspect.signature(fun).parameters)


def call_with(fun, d):
    return fun(**pick(d, params_for(fun)))


def init_with(cls, d, *args, **kwargs):
    return cls(*args, **pick(d, params_for(cls.__init__)), **kwargs)


def notify(title, text='', duration=5000):
    import subprocess as s
    s.call(['notify-send', title, text])


import contextlib
@contextlib.contextmanager
def noop():
    yield None


def pad(x, pad_width, axis=0):
    padding = [[0, 0] for _ in range(x.ndim)]
    padding[axis][1] = pad_width
    return np.pad(x, padding, mode='constant')
