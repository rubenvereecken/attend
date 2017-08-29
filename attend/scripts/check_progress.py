#!/usr/bin/env python

import os
import sys
import argparse
import shutil

from admin import latest_checkpoint_step


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dirs', nargs='*')
    args = parser.parse_args()
    dirs = list(filter(lambda d: not os.path.islink(d) and os.path.isdir(d), args.dirs))

    steps = list(map(latest_checkpoint_step, dirs))

    for dir, step in zip(dirs, steps):
        print('{}\t{}'.format(step, dir))


if __name__ == '__main__':
    main()

