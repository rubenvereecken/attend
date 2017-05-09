#!/usr/bin/env python

# Quick one-off script to gather all annotations in mat format
# and combine them into a navigable hf5 file


import glob
import h5py
import numpy as np
import multiprocessing
import argparse
import os
import shutil

from scipy.io.matlab import loadmat
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
parser.add_argument('-i', '--in_dir', type=str, help='input frames directory, containing subfolders')
parser.add_argument('-o', '--out', type=str, default='annot.hf5',
                    help='file to write to')
# parser.add_argument('-f', '--format', type=str, choices=['hdf5', 'single_hdf5', 'npz', 'np'],
#                     default='hdf5', help='Output format')

def _rename(name):
    if 'annotation_' in name:
        name = name.replace('annotation_', '')
    else:
        name = name.replace('_', '_aligned_')

    return name

def _extract(name):
    parts = name.replace('.mat', '').split('_')
    target, feat = parts[0], parts[-1]

    if 'annotation_' in name:
        extra = 'mean'
    else:
        extra = 'aligned'

    return feat, target, extra


def process_dir(params):
    in_dir = params['in_dir']
    out_file = params['out_file']
    vid_name = params['vid_name']

    mat_files = glob.glob(in_dir + '/*.mat')

    with h5py.File(out_file) as out:
        # annot = out.require_group('annot')
        # Just write to root man
        annot = out

        for mat_file in mat_files:
            # annot_name = _rename(mat_file)
            parts = _extract(mat_file.split('/')[-1])

            target = annot.require_group(parts[1])
            feat = target.require_group(parts[0])
            extra = feat.require_group(parts[2])

            f = loadmat(mat_file)
            data_key = [key for key in f.keys() if not key.startswith('__')][0]

            # For some annoying reason Matlab has it stored as a 1 x N vector
            # Flatten it into a numpy N, vector
            data = f[data_key].flatten()

            dset = extra.create_dataset(vid_name, data=data)


if __name__ == '__main__':
    args = parser.parse_args()


    def _argify(in_dir):
        basename = in_dir.split('/')[-1]
        # out_file = '{}/{}'.format(args.out_dir, args.)

        params = {
            'in_dir': in_dir,
            'out_file': args.out,
            'vid_name': basename,
        }
        return params

    all_args = list(map(_argify, glob.glob(args.in_dir + '/*')))

    for args in tqdm(all_args):
        process_dir(args)

