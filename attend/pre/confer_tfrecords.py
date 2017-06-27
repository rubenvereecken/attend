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



parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
parser.add_argument('-i', '--in_dir', type=str,
                    help='input frames directory, containing subfolders')
parser.add_argument('-o', '--out_dir', type=str, default='out',
                    help='dir to write hf5 files to')
parser.add_argument('--train_frac', type=float, default=2./3,
                    help='Fraction size of training set (< 1)')
parser.add_argument('-a', '--annot_dir', type=str, default=None, required=True,
                    help='Annotation directory')
parser.add_argument('--debug', dest='debug', action='store_true')

parser.set_defaults(debug=False)


def _vid_name_from_dir(d):
    return d.split('/')[-2]


def _split_vids_in_sets(vid_names, train_frac):
    if not isinstance(vid_names, list):
        vid_names = glob.glob(vid_names + '/*')
    n_vids = len(vid_names)
    n_train_vids = math.ceil(train_frac * n_vids)
    np.random.shuffle(vid_names)

    return vid_names[:n_train_vids], vid_names[n_train_vids:]

def process_vids(
        vid_dirs,
        annot_dir,
        out_dir,
        filename,
        debug=False
        ):

    assert len(vid_dirs) != 0, 'Could not find vids'

    def _do_frame(frame_file):
        img = imread(frame_file)
        bbox, bbox_pts, pts = face.extract_bbox(img, mode='face_detect')
        return bbox[0]

    import faceKit as face
    import tensorflow as tf

    def _float32_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _float32_featurelist(value):
        return tf.train.FeatureList(feature=list(map(_float32_feature, value)))
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    writer = tf.python_io.TFRecordWriter(out_dir + '/' + filename + '.tfrecords')

    for vid_dir in (vid_dirs):
        vid_name = _vid_name_from_dir(vid_dir)
        subject_name = vid_dir.split('/')[-1]
        frame_names = sorted(glob.glob(vid_dir + '/*jpg'))
        n_frames = min(len(frame_names), 100) if debug else len(frame_names)
        frame_gen = tqdm((_do_frame(frame) for frame in frame_names[:n_frames]), total=n_frames)

        # In debug mode, keep frames in a list so we can save them
        if debug:
            frames = list(frame_gen)
        else:
            frames = frame_gen

        # Fetch all associated annotations
        m = loadmat(annot_dir + '/' + vid_name + '/meanAnnotation.mat')
        annot = m['annotations'].reshape(-1)[:n_frames]

        features = dict(
                images=tf.train.FeatureList(feature=list(
                    map(_bytes_feature,
                    map(lambda f: f.tobytes(), frames)))),
                conflict=_float32_featurelist(annot)
                )

        example = tf.train.SequenceExample(
                feature_lists=tf.train.FeatureLists(feature_list=features))

        writer.write(example.SerializeToString())

        if debug:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                vid_out_dir = out_dir + '/' + subject_name
                os.makedirs(vid_out_dir)
                for frame_name, frame in zip(frame_names, frames):
                    # frame_name = ''.join(frame_name.split('.')[:-1])
                    frame_name = frame_name.split('/')[-1]
                    imsave(vid_out_dir + '/' + frame_name, frame)

    writer.close()


def _find_deepest_dirs(path='.'):
    for dir, subdirs, files in os.walk(path):
        if len(subdirs) == 0:
            yield dir


if __name__ == '__main__':
    args = parser.parse_args()

    if args.out_dir != '.':
        shutil.rmtree(args.out_dir, ignore_errors=True)

    os.makedirs(args.out_dir, exist_ok=True)

    vid_dirs = list(_find_deepest_dirs(args.in_dir))
    train_vids, test_vids = _split_vids_in_sets(vid_dirs, args.train_frac)
    print(len(train_vids), 'train vids ({:.3f}%)'.format(
        len(train_vids) / (len(train_vids) + len(test_vids))))
    print(len(test_vids), 'test vids')

    train_args = (train_vids[:], args.annot_dir, args.out_dir, 'train')
    test_args = (test_vids[:], args.annot_dir, args.out_dir, 'test')

    run_kwargs = dict(debug=args.debug)
    # run_kwargs['annot_file'] = args.annot_file

    process_vids(*train_args, **run_kwargs)
    process_vids(*test_args, **run_kwargs)

