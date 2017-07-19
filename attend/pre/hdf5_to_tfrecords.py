#!/usr/bin/env python

import math
import glob
import h5py
from itertools import *
from functools import *
import numpy as np
import argparse
import os
import shutil

from tqdm import tqdm

import tensorflow as tf

import attend.pre.util as util


parser = argparse.ArgumentParser(description='blub')
parser.add_argument('-i', '--in_file', type=str,
                    help='input frames directory, containing subfolders')
parser.add_argument('-o', '--out', type=str, default=None,
                    help='dir to write hf5 files to')

parser.set_defaults(debug=False)


def do_sequence_dataset(dset, key=None):
    if key is None:
        key = dset.name.split('/')[-1]

def _intersect_all(sets):
    return reduce(lambda a, b: a.intersection(b), sets[1:], sets[0])

def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float32_featurelist(value):
    return tf.train.FeatureList(feature=list(map(_float32_feature, value)))
def _int64_feature(value):
    if not type(value) == list:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _int64_featurelist(value):
    return tf.train.FeatureList(feature=list(map(_int64_feature, value)))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def process(
        in_file, out_file
        ):
    with h5py.File(in_file, 'r') as f:
        # Assume each group contains sequences and dataset names in groups match
        # if they target the same sequence
        groups = list(f.values())
        group_keys = [set(g.keys()) for g in groups]
        common_keys = _intersect_all(group_keys)

        tfrecord_writer = tf.python_io.TFRecordWriter(out_file)

        for key in tqdm(common_keys):
            ctx = dict(key=_bytes_feature(key.encode('utf8')))
            features = {}
            for group_name, group in f.items():
                dset = group[key]
                arr = dset.value.ravel() # tfrecord only does flat
                if np.issubdtype(dset.dtype, np.float):
                    feature = _float32_featurelist(arr)
                else:
                    raise Exception('{} not supported'.format(dset.dtype))
                # print('{} | {} | final {}'.format(group_name, dset.shape, arr.shape))

                features[group_name] = feature
                # TODO tuple like context feature
                ctx['{}.shape'.format(group_name)] = _int64_feature(list(dset.shape))
                ctx['num_frames'] = _int64_feature(dset.shape[0])

            example = tf.train.SequenceExample(
                    context=tf.train.Features(feature=ctx),
                    feature_lists=tf.train.FeatureLists(feature_list=features)
                )
            tfrecord_writer.write(example.SerializeToString())


def main():
    args = parser.parse_args()

    if args.out is None:
        args.out = args.in_file.replace('.hdf5', '.tfrecords')

    is_file = '.' in args.out.split('/')[-1]
    assert is_file, 'Output must be file'

    util.rm_if_needed(args.out)
    util.makedirs_if_needed(args.out)

    in_ext = args.in_file.split('.')[-1]
    assert in_ext in ['hdf5'], 'Unsupported format {}'.format(in_ext)

    process(args.in_file, args.out)



if __name__ == '__main__':
    main()
