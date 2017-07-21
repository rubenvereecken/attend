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
parser.add_argument('-i', '--in_file', type=str,
                    help='')
parser.add_argument('-o', '--out', type=str, required=True,
                    help='dir to write hf5 files to')
parser.add_argument('--train_frac', type=float, default=0.8)
parser.add_argument('--validation_frac', type=float, default=0.2)
parser.add_argument('--test_frac', type=float, default=0.)

parser.set_defaults(debug=False)


def do_sequence_dataset(dset, key=None):
    if key is None:
        key = dset.name.split('/')[-1]

def _intersect_all(sets):
    return reduce(lambda a, b: a.intersection(b), sets[1:], sets[0])

def extract(
        in_file, out_file,
        keys
        ):
    out = h5py.File(out_file, 'w')
    with h5py.File(in_file, 'r') as f:
        # Assume each group contains sequences and dataset names in groups match
        # if they target the same sequence
        groups = list(f.values())

        for key in tqdm(keys):
            features = {}
            for group_name, group in f.items():
                out_group = out.require_group(group_name)
                out_group.create_dataset(key, data=group[key])
    out.close()

def process(in_file, out_dir, split_names, split_fracs):
    assert len(split_names) == len(split_fracs)
    assert len(split_names) > 0

    num_splits = len(split_names)

    with h5py.File(in_file, 'r') as f:
        groups = list(f.values())
        group_keys = [set(g.keys()) for g in groups]
        common_keys = _intersect_all(group_keys)

    common_keys = sorted(common_keys)
    n = len(common_keys)
    logging.info('Found {} keys to split'.format(len(common_keys)))

    split_sizes = []
    for frac in split_fracs:
        this_size = np.floor(n * frac).astype(int)
        split_sizes.append(this_size)
    leftover = n - sum(split_sizes)

    assert leftover <= num_splits
    for i in range(leftover):
        split_sizes[i] += 1
        leftover -= 1
    assert leftover == 0

    split_keys = []

    offset = 0
    for size in split_sizes:
        split_keys.append(common_keys[offset:offset+size])
        offset += size

    for split_name, keys in zip(split_names, split_keys):
        out_file = '{}/{}.hdf5'.format(out_dir, split_name)
        print('extracting `{}` set ({})'.format(split_name, len(keys)))
        extract(in_file, out_file, keys)



def main():
    args = parser.parse_args()

    if args.out != '.':
        shutil.rmtree(args.out, ignore_errors=True)
    util.makedirs_if_needed(args.out)
    print(args.out)

    in_ext = args.in_file.split('.')[-1]
    assert in_ext in ['hdf5'], 'Unsupported format {}'.format(in_ext)

    split_names = ['train', 'val', 'test']
    split_fracs = [args.train_frac, args.validation_frac, args.test_frac]
    assert sum(split_fracs) == 1., 'Fractions should sum to 1'

    splits = filter(lambda x: x[1] != 0, zip(split_names, split_fracs))
    split_names, split_fracs = zip(*splits)

    process(args.in_file, args.out, split_names, split_fracs)


if __name__ == '__main__':
    main()

