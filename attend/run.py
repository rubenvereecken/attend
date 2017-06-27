#!/usr/bin/env python

import argparse
import inspect
import os
import time

import simplejson as json

from defaults import Defaults
from model import setup_model
from util import pick

parser = argparse.ArgumentParser(description='Run tests')
# parser.add_argument('modality', type=str, choices=['A', 'V', 'AV'],
#         default='AV')
# parser.add_argument('annotation', type=str, choices='valence', 'arousal')
parser.add_argument('-i', '--instances')
# Have the annotations separate so we can switch other ones in
parser.add_argument('-a', '--annotations')
parser.add_argument('--log_dir', type=str, default='logs',
                    help='Directory to hold logs')
parser.add_argument('--no_gen_log_dir', dest='gen_log_dir',
                    action='store_false',
                    help="Don't generate timestamped log dir inside `log_dir`" + \
                         " (default True)")
parser.add_argument('--target', choices=['arousal', 'valence'],
        default='arousal')
parser.add_argument('--data_file', type=str, required=True)

parser.set_defaults(
        gen_log_dir=True,
        **{k: v for k, v in Defaults.__dict__.items() if not k.startswith('__')}
        )


# Push instances through rescaling pipeline
# Load annotations
# Possibly separate script to convert annots from mat to hf5
# Load pretrained network
# Define keras training loop
# Define train/test sets

def _gen_log_dir(base_dir):
    # Create a folder to hold results
    time_str = time.strftime("%d-%m-%Y-%H-%M-%S", time.gmtime())
    base_log_dir = time_str
    log_dir = base_dir + '/' + base_log_dir
    os.makedirs(log_dir, exist_ok=True)

    link_path = base_dir + '/last'

    try:
        # Clean up old softlink
        os.remove(link_path)
    except OSError as ex:
        pass # All G

    os.symlink(base_log_dir, link_path)

    return log_dir


def _save_arguments(args, log_dir):
    file_name = log_dir + '/params.json'
    with open(file_name, 'w') as f:
        json.dump(args.__dict__, f, sort_keys=True, indent='\t')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.gen_log_dir:
        log_dir = _gen_log_dir(args.log_dir)
    else:
        log_dir = args.log_dir
        os.makedirs(log_dir, exist_ok=True)

    _save_arguments(args, log_dir)

    # TODO set up a logging module to be used, global-like

    # This nifty bit takes all the relevant cmd linen args and passes them on
    model = setup_model(**pick(args.__dict__, list(inspect.signature(setup_model).parameters)))
