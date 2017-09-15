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
import logging

from tqdm import tqdm

import tensorflow as tf

import attend.pre.util as util


parser = argparse.ArgumentParser(description='blub')
parser.add_argument('in_file', type=str,
                    help='')
parser.add_argument('-o', '--out', type=str, help='out file')
parser.add_argument('--splits', type=str, required=True,
                    help="Format 'pts1=12 pts2=12'")

parser.set_defaults(debug=False)


def do_sequence_dataset(dset, key=None):
    if key is None:
        key = dset.name.split('/')[-1]

def _intersect_all(sets):
    return reduce(lambda a, b: a.intersection(b), sets[1:], sets[0])

def process(in_file, out_file, splits,
            features_key='features', split_axis=1):
    splits_n = sum(value for key, value in splits)

    out = h5py.File(out_file)
    out_features = out.require_group(features_key)

    with h5py.File(in_file, 'r') as f:
        features = f[features_key]

        for key, dset in tqdm(features.items()):
            original_n = dset.shape[split_axis]
            assert original_n == splits_n, 'Splits must total original size'

            offset = 0
            slices = [slice(None)] * len(dset.shape)
            for split_key, split_n in splits:
                split_group = out_features.require_group(split_key)
                slices[split_axis] = slice(offset, offset + split_n)
                split_group.create_dataset(key, data=dset.value[slices])
                offset += split_n

        # Copy over the rest, might as well
        other_keys = [key for key in f.keys() if key != features_key]
        for key in other_keys:
            f.copy(key, out)

    out.close()


def main():
    args = parser.parse_args()

    if args.out is None:
        basename = os.path.basename(args.in_file)
        dirname = os.path.dirname(args.in_file)
        args.out = dirname + '/' + '.'.join(basename.split('.')[:-1]) + '-split' + '.hdf5'

    util.rm_if_needed(args.out)
    util.makedirs_if_needed(args.out)

    def _split_key_value_pair(s):
        key, value = s.split('=')
        return key, int(value)

    splits = list(map(_split_key_value_pair, args.splits.split(' ')))
    print(splits)

    process(args.in_file, args.out, splits)


if __name__ == '__main__':
    main()

