#!/usr/bin/env python

import argparse
# import inspect
import os
import sys

import simplejson as json

from attend.util import *
import attend


def _save_pid(log_dir):
    file_name = log_dir + '/pid'
    with open(file_name, 'w') as f:
        f.write(str(os.getpid()))


if __name__ == '__main__':
    runner = attend.Runner()
    runner.parse_args_and_setup()
    runner.save_interesting_things()

    all_args = runner.args.__dict__.copy()

    from attend.log import Log
    log = Log.get_logger(__name__)
    if not all_args['debug']:
        log.info('NOT running in debug mode!')

    # Fall back to train batch size
    all_args['val_batch_size'] = all_args['val_batch_size'] or all_args['batch_size']

    import inspect
    from attend.solver import AttendSolver
    from attend.model import AttendModel
    from attend.provider import FileProvider, Provider
    from attend.encoder import Encoder

    all_args['filenames'] = [all_args['data_file']]
    # Bit annoying to create the encoder separately, but it's needed by both
    # the provider (for dimensionality deduction) and the model
    # The goal is to have the provider set up asap so it can start reading
    # all_args['provider'] = Provider(**pick(all_args, params_for(Provider.__init__)))
    all_args['encoder'] = init_with(Encoder, all_args)

    provider_args = all_args.copy()
    all_args['provider'] = FileProvider(all_args['filenames'],
            **pick(all_args, params_for(Provider.__init__)))

    if not runner.args.val_data is None:
        assert os.path.exists(runner.args.val_data), "Validation data not found"
        val_args = all_args.copy()
        # val_args['filenames'] = [runner.args.val_data]
        val_args['shuffle_examples'] = False # Don't shuffle validation data
        val_args['batch_size'] = all_args['val_batch_size']
        all_args['val_provider'] = FileProvider([runner.args.val_data],
                **pick(val_args, params_for(Provider.__init__)))
    all_args['model'] = AttendModel(**pick(all_args, params_for(AttendModel.__init__)))

    solver = init_with(AttendSolver, all_args)
    solver.train(**pick(all_args, list(inspect.signature(solver.train).parameters)))
