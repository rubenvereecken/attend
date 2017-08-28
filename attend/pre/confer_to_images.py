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

from attend.pre.util import *


parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
parser.add_argument('-i', '--in_dir', type=str, required=True,
                    help='input frames directory, containing subfolders')
parser.add_argument('-o', '--out_dir', type=str, default='out/confer-images',
                    help='dir to write output images to')
parser.add_argument('--max_frames', type=int)
parser.add_argument('--debug', dest='debug', action='store_true')

parser.set_defaults(debug=False)


def _vid_name_from_dir(d):
    return d.split('/')[-2]


def process_vids(
        vid_dirs,
        out_dir,
        max_frames=None,
        debug=False
        ):

    if debug:
        debug_out_dir = out_dir + '-debug'
        debug_input_dir = debug_out_dir + '/input'
        debug_transformed_dir = debug_out_dir + '/transformed'
        shutil.rmtree(debug_out_dir, ignore_errors=True)
        os.makedirs(debug_input_dir)
        os.makedirs(debug_transformed_dir)

    from attend.pre import face, util

    def _do_frame(frame_file):
        frame_name = frame_file.split('/')[-1]
        img = imread(frame_file)
        pts_file = frame_file.replace('.jpg', '.pts')
        pts = util.load_pts(pts_file)
        if debug:
            face.paint_and_save(img, pts, debug_input_dir, frame_name)

        try:
            # This is just face.preprocess_and_extract_face's body
            pts_transformed, img_transformed = face.warp_to_mean_shape(pts, img)
            img_cropped = face.extract_face(img_transformed, pts_transformed)
            if 0 in img_cropped.shape:
                raise Exception('Cropped img with a 0 dimension')
            if debug:
                face.paint_and_save(img_transformed, pts_transformed, debug_transformed_dir, frame_name)
            bbox = face.resize(img_cropped)
        except Exception as e:
            print('Failed img {}'.format(frame_file))
            raise e
            # raise e
            bbox = np.zeros((224,224,3))
        return bbox

    for vid_dir in tqdm(vid_dirs):
        vid_name = _vid_name_from_dir(vid_dir)
        subject_name = vid_dir.split('/')[-1]
        frame_names = sorted(glob.glob(vid_dir + '/*jpg'))
        n_all_frames = len(frame_names)
        n_frames = min(n_all_frames, 100) if debug else n_all_frames
        if max_frames and n_frames > max_frames:
            print('Skipping {} of length {}'.format(subject_name, n_frames))
            continue
        frame_gen = tqdm((_do_frame(frame) for frame in frame_names[:n_frames]), total=n_frames)

        # In debug mode, keep frames in a list so we can save them
        frames = list(frame_gen)

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            vid_out_dir = out_dir + '/' + subject_name
            os.makedirs(vid_out_dir)
            for frame_name, frame in zip(frame_names, frames):
                # frame_name = ''.join(frame_name.split('.')[:-1])
                frame_name = frame_name.split('/')[-1]
                frame = frame / max(abs(np.min(frame)), np.max(frame))
                # TODO change frame name to PNG so it's lossless
                imsave(vid_out_dir + '/' + frame_name, frame)


def main():
    args = parser.parse_args()

    if not os.path.isdir(args.in_dir):
        raise Exception('Input directory `{}` does not exist'.format(args.in_dir))

    if args.out_dir != '.':
        shutil.rmtree(args.out_dir, ignore_errors=True)

    os.makedirs(args.out_dir, exist_ok=True)

    vid_dirs = list(find_deepest_dirs(args.in_dir))
    assert len(vid_dirs) != 0, 'Could not find any vids'
    print(len(vid_dirs))

    process_vids(vid_dirs, args.out_dir, max_frames=args.max_frames, debug=args.debug)


if __name__ == '__main__':
    main()

