#!/usr/bin/env python

import glob
import h5py
import numpy as np
import multiprocessing
import argparse
import os
import shutil
from skimage.io import imread
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
parser.add_argument('-i', '--in_dir', type=str, help='input frames directory, containing subfolders')
parser.add_argument('-o', '--out_dir', type=str, default='out',
                    help='dir to write hf5 files to')
parser.add_argument('-f', '--format', type=str, choices=['hdf5', 'single_hdf5', 'npz', 'np'],
                    default='hdf5', help='Output format')
parser.add_argument('-p', '--parallel', type=int, default=4, help='# cores to use')
parser.add_argument('-l', '--compression', type=int, default=4,
        help='gzip 0-9, 0 is off')

def resize_to(w, h):
    def _resize_frame(img):
        pass

    return _resize_frame

PREPROCESS = [
    # resize_to(256, 256),
]

def preprocess_frame(frame):
    for preprocessor in PREPROCESS:
        frame = preprocessor(frame)

    return frame

BIG_FILENAME = 'all.hf5'

# def process_dir(in_dir, out_file):
def process_dir(params):
    # pool.imap is annoying in that it won't **kwarg
    in_dir = params['in_dir']
    out_file = params['out_file']
    format = params['format']
    vid_name = params['vid_name']
    compression = params['compression']

    frame_files = glob.glob(in_dir + '/*.jpg')

    def _do_frame(frame_file):
        img = imread(frame_file)
        img = preprocess_frame(img)
        return img

    frames = list(map(_do_frame, frame_files))
    frames = np.stack(frames)

    if format == 'hdf5':
        with h5py.File(out_file + '.hf5') as out:
            out.create_dataset('img', data=frames, dtype=np.uint8)
    elif format == 'npz':
        np.savez_compressed(out_file, img=frames)

    # Each video is a single dataset, chunks are frames
    elif format == 'single_hdf5':
        with h5py.File(out_file) as out:
            vids = out.require_group('vids')
            chunk_shape = (1,) + frames.shape[1:]

            if compression:
                dset = vids.create_dataset(vid_name, data=frames, dtype=np.uint8,
                                            chunks=chunk_shape, compression='gzip',
                                            compression_opts=compression)
            else:
                dset = vids.create_dataset(vid_name, data=frames, dtype=np.uint8,
                                            chunks=chunk_shape)

    else: pass


    # return frames
    return frames.shape


if __name__ == '__main__':
    args = parser.parse_args()

    # shutil.rmtree(args.out_dir, ignore_errors=True)
    os.makedirs(args.out_dir, exist_ok=True)

    def _argify(in_dir):
        basename = in_dir.split('/')[-1]

        if args.format == 'single_hdf5':
            out_file = '{}/{}'.format(args.out_dir, BIG_FILENAME)
        else:
            out_file = '{}/{}'.format(args.out_dir, basename)
        params = {
            'in_dir': in_dir,
            'out_file': out_file,
            'format': args.format,
            'vid_name': basename,
            'compression': args.compression,
        }
        return params

    all_args = list(map(_argify, glob.glob(args.in_dir + '/*')))

    if args.parallel == 1:
        print('Working single-threaded')
        for a in tqdm(all_args):
            process_dir(a)

    else:
        pool = multiprocessing.Pool(4)

        for result in tqdm(pool.imap_unordered(process_dir, all_args), total=len(all_args)):
            pass

        pool.terminate()
        pool.join()
