#!/usr/bin/env python

import math
import glob
import h5py
import numpy as np
import multiprocessing
import argparse
import os
import shutil

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
parser.add_argument('-a', '--annot_file', type=str, default=None,
        help='Annotation hdf5 file')

def resize_to(w, h):
    def _resize_frame(img):
        return skimage.transform.resize(img, (w, h), mode='reflect',
                preserve_range=True)

    return _resize_frame

PREPROCESS = [
    # resize_to(224, 224),
]

def _split_vids_in_sets(in_dir, train_frac):
    vid_names = glob.glob(in_dir + '/*')
    n_vids = len(vid_names)
    n_train_vids = math.ceil(train_frac * n_vids)
    np.random.shuffle(vid_names)

    return vid_names[:n_train_vids], vid_names[n_train_vids:]


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
        annot_file = h5py.File(annot_file, 'r')
        annot_group_paths = find_leaf_group_paths(annot_file)
        # Filter out everything to do with Audio because this is V only
        annot_group_paths = [path for path in annot_group_paths if '/A' not in path]

    write_batch_size = 500
    n_batches = math.ceil(len(all_frames)/write_batch_size)

    done = False

    for batch_i in tqdm(range(n_batches)):
        n_frames = 0

        this_batch_size = write_batch_size if batch_i < n_batches - 1 else \
                len(all_frames) % write_batch_size
        frames = [None] * this_batch_size

        while not done and n_frames < write_batch_size:
            try:
                frame_file, frame = next(frame_gen)
                frames[n_frames] = frame

                frame_i = _frame_nr_from_file(frame_file)
                vid_name = _vid_name_from_file(frame_file)

                # Take the appropriate data slice from the target

                n_frames += 1

            except StopIteration:
                done = True

        offset = batch_i * write_batch_size
        frame_dset[offset: offset + n_frames] = np.stack(frames)

        batch_i += 1

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

    train_vids, test_vids = _split_vids_in_sets(args.in_dir, args.train_frac)
    print(len(train_vids), 'train vids ({:.3f}%)'.format(
        len(train_vids) / (len(train_vids) + len(test_vids))))
    print(len(test_vids), 'test vids')

    train_args = (args.in_dir, train_vids, args.out_dir + '/train', args.compression)
    test_args = (args.in_dir, test_vids, args.out_dir + '/test', args.compression)

    if args.annot_file:
        train_kwargs = { 'annot_file': args.annot_file }
    else:
        train_kwargs = {}

    # process_vids(*train_args)

    train_thread = Thread(target=process_vids, args=train_args, kwargs=train_kwargs)
    train_thread.start()
    test_thread = Thread(target=process_vids, args=test_args, kwargs=train_kwargs)
    test_thread.start()

    train_thread.join()
    test_thread.join()
