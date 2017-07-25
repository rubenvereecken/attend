#!/usr/bin/env python

import argparse
# import inspect
import os
import sys
import time

import simplejson as json

from attend.defaults import Defaults
from attend.util import *


def _gen_log_dir(base_dir, prefix=''):
    # Create a folder to hold results
    time_str = time.strftime("%d-%m-%Y-%H-%M-%S", time.gmtime())
    if prefix == '' or prefix is None:
        base_log_dir = time_str
    else:
        base_log_dir = '{}_{}'.format(prefix, time_str)
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


def _save_pid(log_dir):
    file_name = log_dir + '/pid'
    with open(file_name, 'w') as f:
        f.write(str(os.getpid()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='data_file', required=True)
    parser.add_argument('--val_data', type=str, default=None)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false')
    parser.add_argument('--log_dir', type=str, default='log',
                        help='Directory to hold logs')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--val_batch_size', type=int)
    parser.add_argument('--steps_per_epoch', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--time_steps', type=int)
    parser.add_argument('--conv_impl', type=str)
    parser.add_argument('--attention_impl', type=str)
    parser.add_argument('--encode_hidden_units', type=int)
    parser.add_argument('--shuffle_examples_capacity', type=int)
    parser.add_argument('--dense_spec', type=str)

    def _boolean_argument(name, default=None):
        parser.add_argument('--{}'.format(name), dest=name, action='store_true')
        parser.add_argument('--no_{}'.format(name), dest=name, action='store_false')
        parser.set_defaults(**{name: default})

    _boolean_argument('encode_lstm')
    _boolean_argument('shuffle_examples')
    _boolean_argument('use_dropout')

    parser.add_argument('--dropout', type=float)
    # parser.add_argument('--dropout', type=float)

    # Not ideal but need to know if debug before things start
    debug = '--debug' in sys.argv
    defaults = Defaults(debug)

    # parser.set_defaults(
    #         gen_log_dir=True, debug=False,
    #         **{k: v for k, v in defaults.__dict__.items() if not k.startswith('__')}
    #         )

    d = defaults.__dict__.copy()
    args = parser.parse_args()

    # Process while defaults not set yet
    if args.shuffle_examples is None and not args.shuffle_examples_capacity is None:
        raise ValueError('Shuffle capacity given without --shuffle_examples')
    if args.encode_lstm is None and not args.encode_hidden_units is None:
        raise ValueError('Encode hidden units given without --encode_lstm')

    d.update({ k: v for (k, v) in args.__dict__.items() if v is not None})
    args.__dict__.update(d)
    args.debug = debug

    if args.gen_log_dir:
        args.log_dir = _gen_log_dir(args.log_dir, args.prefix)
    else:
        os.makedirs(args.log_dir, exist_ok=True)

    _save_arguments(args, args.log_dir)
    _save_pid(args.log_dir)

    from attend.log import Log
    Log.setup(args.log_dir + '/log.txt', args.debug)
    log = Log.get_logger(__name__)

    all_args = args.__dict__.copy()

    if not all_args['debug']:
        log.info('NOT running in debug mode!')

    # Fall back to train batch size
    all_args['val_batch_size'] = all_args['val_batch_size'] or all_args['batch_size']

    import inspect
    from attend.solver import AttendSolver
    from attend.model import AttendModel
    from attend.provider import Provider
    from attend.encoder import Encoder

    all_args['filenames'] = [all_args['data_file']]
    # Bit annoying to create the encoder separately, but it's needed by both
    # the provider (for dimensionality deduction) and the model
    # The goal is to have the provider set up asap so it can start reading
    # all_args['provider'] = Provider(**pick(all_args, params_for(Provider.__init__)))
    all_args['encoder'] = init_with(Encoder, all_args)

    provider_args = all_args.copy()
    all_args['provider'] = init_with(Provider, provider_args)

    if not args.val_data is None:
        assert os.path.exists(args.val_data), "Validation data not found"
        val_args = all_args.copy()
        val_args['filenames'] = [args.val_data]
        val_args['shuffle_examples'] = False # Don't shuffle validation data
        val_args['batch_size'] = all_args['val_batch_size']
        all_args['val_provider'] = init_with(Provider, val_args)
    all_args['model'] = AttendModel(**pick(all_args, params_for(AttendModel.__init__)))

    solver = init_with(AttendSolver, all_args)
    solver.train(**pick(all_args, list(inspect.signature(solver.train).parameters)))
