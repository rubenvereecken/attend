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
parser.add_argument('-o', '--out', type=str, default='out/features.hdf5',
                    help='Output file or folder; format will be inferred')
parser.add_argument('-f', '--feature', type=str, required=True, choices=AVAILABLE_MODELS)
parser.add_argument('--debug', dest='debug', action='store_true')
parser.add_argument('--batch_size', type=int, default=32)
# parser.add_argument('--model', type=str, required=True,
#         choice=['resnet50', 'vggface'])
# parser.add_argument('--layer', type=str)

parser.set_defaults(debug=False)

class BatchedFrameGenerator:
    def __init__(self, name, frame_names, batch_size, feature_dim, batch_processor):
        self.name = name
        self.frame_names = frame_names
        self.n_frames = len(frame_names)
        self.n_batches = np.ceil(self.n_frames / batch_size)
        self.batches = util.batch(frame_names, batch_size)
        self.batch_processor = batch_processor
        self.feature_dim = feature_dim

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.batches)
        return self.batch_processor(batch)


def process_vids(
        vid_dirs,
        model_key,
        batch_size,
        debug=False
        ):

    model = model_for(model_key)
    feature_dim = model.output_shape[1:] # cut off batch dim

    def _do_frames(frame_files):
        # Reads the images and stacks them along batch axis
        images = np.stack(map(imread, frame_files))
        return model.predict(images)


    for vid_dir in (vid_dirs):
        # Really the subject name..
        vid_name = vid_dir.split('/')[-1]
        frame_names = sorted(glob.glob(vid_dir + '/*'))

        frame_gen = BatchedFrameGenerator(vid_name, frame_names,
                batch_size, feature_dim, _do_frames)
        yield frame_gen


def save_to_hdf5(vid_gen, out):
    with h5py.File(out) as out:
        features = out.require_group('features')

        for frames_gen in vid_gen:
            vid_name = frames_gen.name
            dset_dims = (frames_gen.n_frames, *frames_gen.feature_dim)
            vid_dset = features.create_dataset(vid_name, dset_dims, dtype=np.float32)
            offset = 0

            for batch in tqdm(frames_gen, total=frames_gen.n_batches):
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

    vid_dirs = list(util.find_deepest_dirs(args.in_dir))
    assert len(vid_dirs) != 0, 'Could not find any vids'
    print('Found {} directories'.format(len(vid_dirs)))

    # Create a generator that generates batches of features per video
    vids = process_vids(vid_dirs, model_key=args.feature,
            debug=args.debug, batch_size=args.batch_size)

    # Write out the batches of features
    writers[out_ext](vids, args.out)


if __name__ == '__main__':
    main()
