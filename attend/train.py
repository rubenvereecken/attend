#!/usr/bin/env python

import argparse
# import inspect
import os
import time

import simplejson as json

from attend.defaults import Defaults
from attend.solver import AttendSolver
from attend.util import *


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
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='data_file')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.add_argument('--log_dir', type=str, default='log',
                        help='Directory to hold logs')
    parser.add_argument('--batch_size', type=int)

    parser.set_defaults(
            gen_log_dir=True, debug=False,
            **{k: v for k, v in Defaults.__dict__.items() if not k.startswith('__')}
            )

    args = parser.parse_args()

    if args.gen_log_dir:
        args.log_dir = _gen_log_dir(args.log_dir)
    else:
        os.makedirs(args.log_dir, exist_ok=True)

    _save_arguments(args, args.log_dir)

    # model = setup_model(**pick(args.__dict__, list(inspect.signature(setup_model).parameters)))
    # solver = SEWASolver(None)
    all_args = args.__dict__.copy()

    if not all_args['debug']:
        print('NOT running in debug mode!')

    import inspect
    from attend.model import AttendModel
    from attend.provider import Provider

    all_args['filenames'] = [all_args['data_file']]
    all_args['provider'] = Provider(**pick(all_args, params_for(Provider.__init__)))
    all_args['model'] = AttendModel(**pick(all_args, params_for(AttendModel.__init__)))
    # all_args['model'] = AttendModel()
    model = all_args['model']

    # The actual training bit
    # solver = AttendSolver(**pick(all_args, list(inspect.signature(AttendSolver.__init__).parameters)))
    solver = AttendSolver(model, update_rule='adam', learning_rate=0.01)
    solver.train(**pick(all_args, list(inspect.signature(solver.train).parameters)))
    # print(pick(all_args, list(inspect.signature(solver.train).parameters)))
    # solver.train(args.data_file, 20, 1, log_dir=args.log_dir)
