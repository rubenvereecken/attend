import tensorflow as tf
import os
import sys


def logdirs_without_checkpoints(path):
    dirs = os.listdir(path)
    dirs = (map(lambda d: path + '/' + d, dirs))
    dirs = list(filter(lambda d: not os.path.islink(d) and os.path.isdir(d), dirs))

    matches = zip(map(lambda d: tf.train.latest_checkpoint(d), dirs), dirs)
    matches = filter(lambda d: d[0] is None, matches)

    return list(zip(*matches))[1]
