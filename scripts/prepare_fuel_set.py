#!/usr/bin/env python

import math
import glob
import h5py
import numpy as np
import multiprocessing
import argparse
import os
import shutil

from fuel.datasets.hdf5 import H5PYDataset
from threading import Thread
from skimage.io import imread
import skimage.transform

from tqdm import tqdm

from util import find_leaf_group_paths

parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
parser.add_argument('-i', '--in_dir', type=str, help='input frames directory, containing subfolders')
parser.add_argument('-o', '--out_dir', type=str, default='out',
                    help='dir to write hf5 files to')
# parser.add_argument('-f', '--format', type=str, choices=['hdf5', 'single_hdf5', 'npz', 'np'],
#                     default='hdf5', help='Output format')
parser.add_argument('--train_frac', type=float, default=2./3,
                    help='Fraction size of training set (< 1)')
parser.add_argument('--resize', type=str,
                    help='x,y resize dimensions - leave empty to not resize')
# parser.add_argument('-p', '--parallel', type=int, default=4, help='# cores to use')
parser.add_argument('-l', '--compression', type=int, default=4,
        help='gzip 0-9, 0 is no compression')
parser.add_argument('-a', '--annot_file', type=str, default=None, required=True,
        help='Annotation hdf5 file')

def resize_to(w, h):
    def _resize_frame(img):
        return skimage.transform.resize(img, (w, h), mode='reflect',
                preserve_range=True)

    return _resize_frame

PREPROCESS = [
    # resize_to(224, 224),
]

def preprocess_frame(frame):
    for preprocessor in PREPROCESS:
        frame = preprocessor(frame)

    return frame


CHUNK_SIZE = 10

def _frame_nr_from_file(fname):
    base = fname.split('/')[-1]
    return int(base.split('.')[0]) - 1

def _vid_name_from_file(fname):
    return fname.split('/')[-2]

def process_vids(
        in_dir,
        vid_names,
        file_name,
        train_frac,
        compression=0,
        annot_file=None,
        shuffle=True):
    all_frames = []

    for vid_name in vid_names:
        all_frames += glob.glob(vid_name + '/*')

    np.random.shuffle(all_frames)

    # Do first frame just to get idea of data shape
    img_shape = preprocess_frame(imread(all_frames[0])).shape

    def _do_frame(frame_file):
        img = imread(frame_file)
        img = preprocess_frame(img)
        return frame_file, img

    frame_gen = map(_do_frame, all_frames)

    hfile = h5py.File(file_name + '.hf5')

    if compression:
        frame_dset = hfile.create_dataset('img',
                shape=(len(all_frames),) + img_shape, dtype=np.uint8,
                chunks=(CHUNK_SIZE,) + img_shape,
                compression='gzip', compression_opts=compression)
    else:
        print('No compression or chunking')
        frame_dset = hfile.create_dataset('img',
                shape=(len(all_frames),) + img_shape, dtype=np.uint8)

    if annot_file:
        print('Including target annotations')
        annot_file = h5py.File(annot_file, 'r')
        annot_group_paths = find_leaf_group_paths(annot_file)
        annot_group_paths = [path for path in annot_group_paths if 'liking' not in path]

    write_batch_size = 500
    n_batches = math.ceil(len(all_frames)/write_batch_size)

    done = False

    for batch_i in tqdm(range(n_batches)):
        n_frames = 0

        this_batch_size = write_batch_size if batch_i < n_batches - 1 else \
                len(all_frames) % write_batch_size
        frames = np.empty((this_batch_size,) + img_shape)
        targets = { k: np.empty((this_batch_size, )) for k in annot_group_paths }

        while not done and n_frames < write_batch_size:
            try:
                frame_file, frame = next(frame_gen)
                frames[n_frames] = frame

                frame_i = _frame_nr_from_file(frame_file)
                vid_name = _vid_name_from_file(frame_file)

                # Take the appropriate data slice from the target
                if annot_file:
                    for annot_group in annot_group_paths:
                        try:
                            targets[annot_group][n_frames] = annot_file[annot_group][vid_name][frame_i]
                        except KeyError as e:
                            print('FAILED')
                            print(annot_group, vid_name, frame_i)
                            print(e)
                            raise e

                n_frames += 1

            except StopIteration:
                done = True

        offset = batch_i * write_batch_size
        frame_dset[offset: offset + n_frames] = frames

        # Write away target batches one by one, per type
        for annot_group in annot_group_paths:
            target_dset = hfile.require_dataset(annot_group,
                    shape=(len(all_frames),), dtype=np.float)
            target_dset[offset: offset + n_frames] = targets[annot_group]

        batch_i += 1

    train_end_i = math.ceil(len(all_frames) * train_frac)
    splits = { 'train': (0, train_end_i), 'test': (train_end_i, len(all_frames)) }
    split_dict = { split_key: { 'img': split_i } for split_key, split_i in splits.items() }
    for split_key, split_i in splits.items():
        for annot_group in annot_group_paths:
            split_dict[split_key][annot_group] = split_i

    hfile.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    hfile.close()
    if annot_file:
        annot_file.close()


if __name__ == '__main__':
    args = parser.parse_args()

    if args.resize:
        resize = tuple(map(int, args.resize.split(',')))

        PREPROCESS.append(resize_to(resize[0], resize[1]))
        print('Also resizing to', resize)

    if args.out_dir != '.':
        shutil.rmtree(args.out_dir, ignore_errors=True)

    os.makedirs(args.out_dir, exist_ok=True)

    all_vids = glob.glob(args.in_dir + '/*')
    # all_vids = all_vids[:2]

    run_args = (args.in_dir, all_vids, args.out_dir + '/all', args.train_frac, args.compression)

    if args.annot_file:
        train_kwargs = { 'annot_file': args.annot_file }
    else:
        train_kwargs = {}

    process_vids(*run_args, **train_kwargs)
