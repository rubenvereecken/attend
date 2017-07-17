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

this_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
parser.add_argument('-i', '--in_dir', type=str, required=True,
                    help='input frames directory, containing subfolders')
parser.add_argument('-o', '--out', type=str, default='out/features.hdf5',
                    help='Output file or folder; format will be inferred')
parser.add_argument('-f', '--feature', type=str, required=True, choices=AVAILABLE_MODELS)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--mean_pixel', type=str, default=this_path + '/data/imagenet_mean_pixel.npy')
parser.add_argument('--dont_subtract_mean_pixel', action='store_false',
        dest='subtract_mean_pixel')

parser.set_defaults(debug=False)
parser.set_defaults(subtract_mean_pixel=True)

def process_vids(
        vid_dirs,
        model_key,
        batch_size,
        mean_pixel=None,
        debug=False
        ):

    model = model_for(model_key)
    feature_dim = model.output_shape[1:] # cut off batch dim
    if not mean_pixel is None:
        # TF works with float32 anyway
        mean_pixel = mean_pixel.astype(np.float32)

    def _do_frame(frame_file):
        image = imread(frame_file).astype(np.float32)
        # Subtract mean pixel if given
        if not mean_pixel is None:
            image -= mean_pixel
        return image

    def _do_frames(frame_files):
        # Reads the images and stacks them along batch axis
        images = np.stack(map(_do_frame, frame_files))
        return model.predict(images)


    def _video_gen():
        for vid_dir in (vid_dirs):
            # Really the subject name..
            vid_name = vid_dir.split('/')[-1]
            frame_names = sorted(glob.glob(vid_dir + '/*'))
            n_batches = np.ceil(len(frame_names) / batch_size).astype(int)

            _frame_gen = (_do_frames(batch) for batch in
                    util.batch(frame_names, batch_size))
            frame_gen = util.LengthyGenerator(_frame_gen, n_batches)
            frame_gen['num_frames'] = len(frame_names)
            frame_gen['feature_dim'] = feature_dim
            frame_gen['name'] = vid_name
            yield frame_gen

    video_gen = util.LengthyGenerator(_video_gen(), len(vid_dirs))
    return video_gen


def save_to_hdf5(vid_gen, out):
    with h5py.File(out) as out:
        features = out.require_group('features')

        for frames_gen in tqdm(vid_gen, desc='video'):
            vid_name = frames_gen['name']
            dset_dims = (frames_gen['num_frames'], *frames_gen['feature_dim'])
            vid_dset = features.create_dataset(vid_name, dset_dims, dtype=np.float32)
            offset = 0

            for batch in tqdm(frames_gen, desc='batch'):
                batch_size = batch.shape[0]
                vid_dset[offset:offset+batch_size] = batch
                offset += batch_size

writers = dict(
        hdf5=save_to_hdf5
        )

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

    # Mean pixel
    if args.subtract_mean_pixel:
        mean_pixel = np.load(args.mean_pixel)
    else:
        print('Not using the mean pixel value')
        mean_pixel = None

    vid_dirs = list(util.find_deepest_dirs(args.in_dir))
    assert len(vid_dirs) != 0, 'Could not find any vids'
    print('Found {} directories'.format(len(vid_dirs)))

    # Create a generator that generates batches of features per video
    vids = process_vids(vid_dirs, model_key=args.feature,
            debug=args.debug, batch_size=args.batch_size, mean_pixel=mean_pixel)

    # Write out the batches of features
    writers[out_ext](vids, args.out)


if __name__ == '__main__':
    main()
