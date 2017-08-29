import tensorflow as tf
import os
import sys


def logdirs_without_checkpoints(path):
    dirs = os.listdir(path)
    dirs = (map(lambda d: path + '/' + d, dirs))
    dirs = list(filter(lambda d: not os.path.islink(d) and os.path.isdir(d), dirs))

    matches = zip(map(lambda d: tf.train.latest_checkpoint(d), dirs), dirs)
    matches = list(filter(lambda d: d[0] is None, matches))

    return list(zip(*matches))[1] if len(matches) > 0 else []


def match_dirs_by_checkpoint_step(dirs, pred):
    dirs = list(filter(lambda d: not os.path.islink(d) and os.path.isdir(d), dirs))

    matches = zip(map(lambda d: latest_checkpoint_step(d), dirs), dirs)
    matches = list(filter(lambda d: pred(d[0]), matches))

    steps_and_dirs = list(zip(*matches)) if len(matches) > 0 else [[],[]]
    return steps_and_dirs[1], steps_and_dirs[0]


def latest_checkpoint_step(log_dir):
    checkpoint = tf.train.latest_checkpoint(log_dir)
    if checkpoint is None:
        return 0
    return int(checkpoint.split('-')[-1])
