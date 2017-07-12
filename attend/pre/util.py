import os

def find_deepest_dirs(path='.'):
    for dir, subdirs, files in os.walk(path):
        if len(subdirs) == 0:
            yield dir


