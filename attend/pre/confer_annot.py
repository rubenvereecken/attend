#!/usr/bin/env python

import math
import glob
import h5py
import itertools
import numpy as np
import multiprocessing
import argparse
import os
import shutil

from threading import Thread
from skimage.io import imread, imsave
from scipy.io.matlab import loadmat
import skimage.transform

from tqdm import tqdm

import attend.pre.util as util
from attend.pre.models import AVAILABLE_MODELS
from attend.pre.models import *


parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
parser.add_argument('-i', '--in_dir', type=str, required=True,
                    help='input frames directory, containing subfolders')
parser.add_argument('-a', '--annot_dir', type=str, required=True,
        help='')
parser.add_argument('-o', '--out', type=str, default='out/annotations.hdf5',
                    help='Output file or folder; format will be inferred')
parser.add_argument('--debug', dest='debug', action='store_true')

parser.set_defaults(debug=False)


def process_annot(
        vid_dirs,
        annot_dir,
        out,
        debug=False
        ):

    with h5py.File(out) as out:
        annotations = out.require_group('conflict')

        for vid_dir in tqdm(vid_dirs):
            # Video directory is vid_name/subject_name
            # While annot directory is just vid_name/, same for both subjects
            vid_name = vid_dir.split('/')[-2]
            subject_name = vid_dir.split('/')[-1]

            # Write annotations per subject name, even though duplicate
            m = loadmat('{}/{}/meanAnnotation.mat'.format(annot_dir, vid_name))
            annot = m['annotations'].reshape(-1)
            annotations.create_dataset(subject_name, data=annot, dtype=np.float32)


def main():
    args = parser.parse_args()

    if not os.path.isdir(args.in_dir):
        raise Exception('Input directory `{}` does not exist'.format(args.in_dir))

    is_file = '.' in args.out.split('/')[-1]
    assert is_file, 'Directory output not supported yet, try file ext'

    util.rm_if_needed(args.out)
    util.makedirs_if_needed(args.out)

    out_ext = args.out.split('.')[-1]
    assert out_ext in ['hdf5'], 'Unsupported format {}'.format(out_ext)

    vid_dirs = list(util.find_deepest_dirs(args.in_dir))
    assert len(vid_dirs) != 0, 'Could not find any vids'

    annot_dirs = list(util.find_deepest_dirs(args.annot_dir))
    assert len(annot_dirs) != 0

    process_annot(vid_dirs, args.annot_dir, args.out, debug=args.debug)

if __name__ == '__main__':
    main()

