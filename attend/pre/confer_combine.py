#!/usr/bin/env python

import numpy as np
import h5py
import argparse
import os
import shutil

from tqdm import tqdm

from attend.pre import util

parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
parser.add_argument('-i', '--in_file', type=str, required=True,
                    help='input frames directory, containing subfolders')
parser.add_argument('-o', '--out', type=str, default=None,
                    help='output hdf5 file')

parser.set_defaults(debug=False)


def _vid_name(s):
    return '_'.join(s.split('_')[:2])

def process(in_file, out_file):
    out = h5py.File(out_file, 'w')
    combined = out.require_group('features')

    with h5py.File(in_file) as f:
        features = f.require_group('features')
        by_vid = {}

        for k, dset in features.items():
            vid_name = _vid_name(k)
            l = by_vid.get(vid_name, [])
            l.append(dset)
            by_vid[vid_name] = l

        for k, dsets in tqdm(by_vid.items()):
            assert len(dsets) == 2
            # This creates an extra dimension at axis 1 of size 2
            final_dset = np.stack(dsets, axis=1)
            combined.create_dataset(k, data=final_dset)

    out.close()


def main():
    args = parser.parse_args()

    if not os.path.isfile(args.in_file):
        raise Exception('Input file `{}` does not exist'.format(args.in_file))

    if args.out is None:
        basename = '.'.join(args.in_file.split('.')[:-1])
        args.out = basename + '-combined.hdf5'

    is_file = '.' in args.out.split('/')[-1]
    assert is_file, 'Directory output not supported yet, try file ext'

    util.rm_if_needed(args.out)
    util.makedirs_if_needed(args.out)

    in_ext = args.in_file.split('.')[-1]
    assert in_ext in ['hdf5'], 'Unsupported format {}'.format(out_ext)

    process(args.in_file, args.out)


if __name__ == '__main__':
    main()
