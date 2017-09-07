#!/usr/bin/env python

import math
import glob
import h5py
from scipy.io import loadmat
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
parser.add_argument('in_dir', type=str,
                    help='')
parser.add_argument('-o', '--out', type=str, required=True ,
                    help='')
parser.add_argument('-k', '--in_key', type=str, default='features')
parser.add_argument('--out_group', type=str, default=None,
                    help='Will default to input key')

parser.set_defaults(debug=False)


def mat_to_array(in_file, key):
    m = loadmat(in_file)
    return m[key]


def process(mat_files, out_file, in_key, out_group=None,
            flip_trailing_dim=True):
    if out_group is None:
        out_group = in_key

    with h5py.File(out_file) as out:
        g = out.require_group(out_group)

        for mat_file in tqdm(mat_files):
            name = ''.join(os.path.basename(mat_file).split(os.path.extsep)[:-1])
            m = loadmat(mat_file)
            arr = m[in_key]
            if flip_trailing_dim:
                arr = np.swapaxes(arr, 0, 1)

            g.create_dataset(name, data=arr)



def main():
    args = parser.parse_args()

    is_file = '.' in args.out.split('/')[-1]
    assert is_file, 'Output must be file'

    util.rm_if_needed(args.out)
    util.makedirs_if_needed(args.out)

    in_files = list(glob.glob(args.in_dir + '/*.mat'))
    assert len(in_files) > 0, 'No mat files found'

    process(in_files, args.out, args.in_key, args.out_group)


if __name__ == '__main__':
    main()

