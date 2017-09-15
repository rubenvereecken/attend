import inspect
import numpy as np
import attend
import time
from functools import partial

# parse_timestamp = partial(time.strptime, format=attend.TIMESTAMP_FORMAT)
parse_timestamp = lambda x: time.strptime(x, attend.TIMESTAMP_FORMAT)

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


def pad_and_stack(arrays, length=None, axis=0):
    if length is None:
        length = max(v.shape[0] for v in arrays)

    def _padding(v):
        padding = np.zeros([len(v.shape), 2], dtype=int)
        assert length >= v.shape[axis]
        padding[axis][1] = length - v.shape[axis]
        return padding

    return np.stack(np.pad(v, _padding(v), 'constant') for v in arrays)


def unstack_and_unpad(arrays, lengths):
    return [arr[:lengths[i]] for i, arr in enumerate(arrays)]


def dict_to_args(d):
    arg_list = []

    for k, v in d.items():
        if v is None:
            # Don't pass None values, just 'none' values
            continue
            # s = 'none'
        elif isinstance(v, bool):
            s = str(int(v)) # Boolean is represented 0 or 1
        elif isinstance(v, list):
            # s = ' '.join(el for el in v)
            s = ' '.join('--{}={}'.format(k, el) for el in v)
            arg_list.append(s)
            continue
        else:
            s = str(v)

        s = '--{}={}'.format(k, s)

        arg_list.append(s)

    return ' '.join(arg_list)

