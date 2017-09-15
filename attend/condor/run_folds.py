#!/usr/bin/env python

import math
import glob
import h5py
from itertools import *
from functools import *
import numpy as np
import argparse
import os
import shutil
import logging

from tqdm import tqdm
import pydash
import cson


parser = argparse.ArgumentParser()
parser.add_argument('folds_dir', type=str)
parser.add_argument('--base_dir', type=str)
parser.add_argument('--prefix', type=str, default='base')


def create_folds(n):
    indices = list(range(n)) + list(range(n-1))

    def _fold_gen():
        for g in zip(*[indices[i:] for i in range(n)]):
            train = g[:-2]
            val = g[-2]
            test = g[-1]

            yield(train, val, test)

    return list(_fold_gen())


def main():
    from attend.pre.util import rm_if_needed
    args, rest_args = parser.parse_known_args()
    folds_files = sorted(glob.glob(args.folds_dir + '/*tfrecords'))
    fold_indices = create_folds(len(folds_files))

    if args.base_dir:
        log_dir = args.base_dir
    else:
        log_dir = '/vol/bitbucket/rv1017/log-' + args.prefix
    rm_if_needed(log_dir, True)

    job_dir = log_dir + '/jobs'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(job_dir, exist_ok=True)

    from attend.util import dict_to_args
    from attend.condor import generate_job

    fold_info = {}

    rest_args += ['--val_batch_size=8', '--keep_all_checkpoints=1']

    for train_idxs, val_idx, test_idx in fold_indices:
        train_idxs = list(train_idxs)
        fold_idx = train_idxs[0] + 1
        train_files = [folds_files[i] for i in train_idxs]
        val_file = folds_files[val_idx]
        test_file = folds_files[test_idx]
        pargs = dict(data_file=train_files, val_data=val_file)
        pargs_str = dict_to_args(pargs)
        job_prefix = '{}.{}'.format(args.prefix, fold_idx)

        job_desc = generate_job(job_prefix,
                                rest_args=pargs_str.split(' ') + rest_args,
                                base_log_path=log_dir)

        with open('{}/{}.classad'.format(job_dir, fold_idx), 'w') as f:
            f.write(job_desc)

        # fold_info[fold_idx] = (dict(train_files=train_files, val_file=val_file,
        #                             test_file=test_file, prefix=job_prefix))
        fold_info[fold_idx] = {}
        fold_info[fold_idx].update(dict(prefix=job_prefix,
            train_idx=train_idxs, val_idx=val_idx, test_idx=test_idx))

    with open(log_dir + '/folds.cson', 'w') as f:
        cson.dump(fold_info, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    main()
