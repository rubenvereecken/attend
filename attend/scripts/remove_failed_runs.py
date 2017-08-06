#!/usr/bin/env python

import os
import sys
import argparse
import shutil

from admin import logdirs_without_checkpoints


def main():
    parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
    parser.add_argument('in_dir', type=str)
    parser.add_argument('-n', '--dry-run', action='store_true')
    parser.set_defaults(dry_run=False)
    args = parser.parse_args()

    if not os.path.exists(args.in_dir):
        raise Exception('Directory does not exist')

    dirs = logdirs_without_checkpoints(args.in_dir)

    if args.dry_run:
        print('{} directories to be removed:'.format(len(dirs)))
        for d in dirs:
            print(d)
        sys.exit(0)


    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

    print('Removed {} directories'.format(len(dirs)))



if __name__ == '__main__':
    main()
