import argparse
import os
import sys

from attend.defaults import Defaults
from attend.log import Log

class Runner:

    def __init__(self):
        self.debug = '--debug' in sys.argv
        self.parser = self._setup_parser()


    def _setup_parser(self):
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
        _boolean_argument('use_maxnorm')

        parser.add_argument('--dropout', type=float)

        return parser


    def parse_args_and_setup(self):
        defaults = Defaults(self.debug).__dict__
        args = self.parser.parse_args()
        self.args = args

        # Process while defaults not set yet
        if args.shuffle_examples is None and not args.shuffle_examples_capacity is None:
            raise ValueError('Shuffle capacity given without --shuffle_examples')
        if args.encode_lstm is None and not args.encode_hidden_units is None:
            raise ValueError('Encode hidden units given without --encode_lstm')

        defaults.update({ k: v for (k, v) in args.__dict__.items() if v is not None})
        args.__dict__.update(defaults)
        args.debug = self.debug

        if args.gen_log_dir:
            args.log_dir = self._gen_log_dir(args.log_dir, args.prefix)
        else:
            os.makedirs(args.log_dir, exist_ok=True)

        self.setup_log(args.log_dir)
        log = Log.get_logger(__name__)

        if 'LD_PRELOAD' in os.environ and 'tcmalloc' in os.environ['LD_PRELOAD']:
            log.debug('Using tcmalloc, good on you!')
        else:
            log.warning('NOT using tcmalloc')


    def setup_log(self, log_dir=None):
        if log_dir is None:
            log_dir = self.args.log_dir
        from attend.log import Log
        Log.setup(log_dir, self.debug)


    def save_interesting_things(self):
        Log.save_args(self.args)
        Log.save_pid()
        Log.save_env()
        Log.save_git_version()


    def _gen_log_dir(self, base_dir, prefix=''):
        # Create a folder to hold results
        import time
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
