#!/usr/bin/env python

import os
import sys
import argparse
import shutil
import math

from admin import match_dirs_by_checkpoint_step


def _build_predicate(lte, gte):
    if gte is None:
        gte = math.inf
    if lte is None:
        lte = -1

    return lambda x: x <= lte or x >= gte


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dirs', nargs='*', type=str)
    parser.add_argument('-n', '--dry-run', action='store_true')
    parser.add_argument('-c', '--confirm', action='store_true')
    parser.add_argument('--lte', type=int)
    parser.add_argument('--gte', type=int)
    parser.set_defaults(dry_run=False, confirm=False)
    args = parser.parse_args()

    # if not os.path.exists(args.in_dir):
    #     raise Exception('Directory does not exist')

    if args.lte is None and args.gte is None:
        print('No criteria provided, defaulting to removing failed runs')
        args.lte = 0

    pred = _build_predicate(args.lte, args.gte)
    dirs, steps = match_dirs_by_checkpoint_step(args.dirs, pred)

    if len(dirs) == 0:
        print('No directories to be removed')
        sys.exit(0)

    def _print_dirs():
        print('{} directories to be removed:'.format(len(dirs)))
        print('step\tdir')
        print('----\t----')
        for d, step in zip(dirs, steps):
            print('{}\t{}'.format(step, d))

    if args.dry_run:
        _print_dirs()
        sys.exit(0)

    if args.confirm:
        _print_dirs()
        y = input('Remove? (y/N) ')
        if y.strip() != 'y':
            sys.exit(0)

    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

    print('Removed {} directories'.format(len(dirs)))



if __name__ == '__main__':
    main()

