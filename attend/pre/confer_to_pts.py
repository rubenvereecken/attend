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

from attend.pre import face, util


parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
parser.add_argument('-i', '--in_dir', type=str, required=True,
                    help='input frames directory, containing subfolders')
parser.add_argument('-o', '--out', type=str, required=True,
                    help='dir to write output images to')
parser.add_argument('--debug', dest='debug', action='store_true')

parser.set_defaults(debug=False)


def _vid_name_from_dir(d):
    return d.split('/')[-2]


def process_vids(
        vid_dirs,
        out_file,
        debug=False
        ):


    vid_dirs_by_vid = {}

    for vid_dir in (vid_dirs):
        vid_name = _vid_name_from_dir(vid_dir)
        l = vid_dirs_by_vid.get(vid_name, [])
        l.append(vid_dir)
        vid_dirs_by_vid[vid_name] = l

    def _do_pts(pts_file):
        pts = util.load_pts(pts_file)
        pts = face.warp_to_mean_shape(pts)
        pts = face.mean_and_stdev_normalize(pts)
        return pts

    def _do_vid(vid_dir):
        subject_name = vid_dir.split('/')[-1]
        frame_names = sorted(glob.glob(vid_dir + '/*pts'))
        assert len(frame_names) != 0
        vid_pts = np.stack(tqdm((_do_pts(frame) for frame in frame_names),
            total=len(frame_names)))
        return vid_pts

    assert min(map(lambda d: len(d), vid_dirs_by_vid.values())) > 1

    with h5py.File(out_file) as out:
        features = out.require_group('features')
        for vid_name, vid_dirs in tqdm(vid_dirs_by_vid.items()):
            # Concatenate points, is now T x 2*68 x 2
            # Expects vids to be of equal length
            all_pts = np.concatenate(list(_do_vid(vids) for vids in vid_dirs), axis=1)
            vid_dset = features.create_dataset(vid_name, data=all_pts)


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
    print('Found {} directories'.format(len(vid_dirs)))

    process_vids(vid_dirs, args.out)


if __name__ == '__main__':
    main()


