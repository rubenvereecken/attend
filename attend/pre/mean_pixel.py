#!/usr/bin/env python

import glob
import itertools
import numpy as np
import argparse
import os

from skimage.io import imread, imsave

from tqdm import tqdm

import attend.pre.util as util

parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
parser.add_argument('-i', '--in_dir', type=str, required=True,
                    help='input frames directory, containing subfolders')
parser.add_argument('-o', '--out', type=str, default='out/mean_pixel.npy',
                    help='Output file or folder; format will be inferred')


def process_vids(vid_dirs):
    def _do_frame(frame_file):
        img = imread(frame_file)
        return np.mean(img, (0,1))

    def _do_frames(frame_files):
        means = np.stack(map(_do_frame, tqdm(frame_files)))
        return np.mean(means, 0)

    def _vid_gen():
        for vid_dir in tqdm(vid_dirs):
            # Really the subject name..
            vid_name = vid_dir.split('/')[-1]
            frame_names = sorted(glob.glob(vid_dir + '/*'))
            frame_names = list(frame_names)

            this_pixel = _do_frames(frame_names)
            assert this_pixel.shape == (3,)
            yield this_pixel

    means = np.stack(_vid_gen())
    mean_pixel = np.mean(means, 0)
    return mean_pixel


def save_to_npy(pixel, out):
    np.save(out, pixel)


writers = dict(
        npy=save_to_npy
        )


def main():
    args = parser.parse_args()
    if not os.path.isdir(args.in_dir):
        raise Exception('Input directory `{}` does not exist'.format(args.in_dir))

    is_file = '.' in args.out.split('/')[-1]
    assert is_file

    util.rm_if_needed(args.out)
    util.makedirs_if_needed(args.out)

    out_ext = args.out.split('.')[-1]
    assert out_ext in ['npy'], 'Unsupported format {}'.format(out_ext)

    vid_dirs = list(util.find_deepest_dirs(args.in_dir))
    assert len(vid_dirs) != 0, 'Could not find any vids'
    print('Found {} directories'.format(len(vid_dirs)))

    # Create a generator that generates batches of features per video
    pixel = process_vids(vid_dirs)

    # Write out the batches of features
    writers[out_ext](pixel, args.out)


if __name__ == '__main__':
    main()
