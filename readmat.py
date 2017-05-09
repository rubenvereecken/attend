#!/usr/bin/env python
# Convenience script to open .mat files with
# Usage:
# python -i readmat.py $1

from scipy.io.matlab import loadmat
import sys

if __name__ == '__main__':
    m = loadmat(sys.argv[1])
    print(m)
