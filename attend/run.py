import argparse
import os
import sys

from attend.defaults import Defaults
from attend.log import Log

class Runner:

    def __init__(self):
        self.debug = '--debug' in sys.argv
        self.defaults = Defaults(self.debug).__dict__
        self.parser = self._setup_parser()


    def _setup_parser(self):
        parser = argparse.ArgumentParser()

        # Booleans are passed with a 0 or 1 on the CLI for easy scripting
        self.boolean_args = []
        def _boolean_argument(name, parser=parser, default=None):
            # parser.add_argument('--{}'.format(name), dest=name, action='store_true')
            parser.add_argument('--no_{}'.format(name), dest=name, action='store_const', const=0)
            parser.add_argument('--{}'.format(name), dest=name, type=int)
            # parser.add_argument('--{}'.format(name), dest=name, type=int)
            self.boolean_args.append(name)
            parser.set_defaults(**{name: default})

        logistics = parser.add_argument_group('Logistics')
        # nargs='*' doesn't work if you use '='
        logistics.add_argument('-i', '--data_file', dest='data_file', required=True,
                               action='append')
        logistics.add_argument('--val_data', type=str, default=None)
        logistics.add_argument('--debug', dest='debug', action='store_true')
        logistics.add_argument('--log_dir', type=str, default='log',
                            help='Directory to hold logs')
        logistics.add_argument('--prefix', type=str, default='')
        _boolean_argument('save_eval_graph', logistics)
        _boolean_argument('gen_log_dir', logistics)
        _boolean_argument('show_progress_bar', logistics)
        _boolean_argument('restore_if_possible', logistics)
        _boolean_argument('restore_or_die', logistics)
        _boolean_argument('keep_all_checkpoints', logistics)

        experiment = parser.add_argument_group('Experiment')
        experiment.add_argument('--steps_per_epoch', type=int)
        experiment.add_argument('--num_epochs', type=int)
        experiment.add_argument('--stats_every', type=int)

        input = parser.add_argument_group('Input feeding')
        input.add_argument('--batch_size', type=int)
        input.add_argument('--val_batch_size', type=int)
        input.add_argument('--shuffle_examples_capacity', type=int)
        _boolean_argument('shuffle_examples', input)

        training = parser.add_argument_group('Training')
        training.add_argument('--update_rule', type=str)
        training.add_argument('--learning_rate', type=float)
        training.add_argument('--loss_function', type=str)

        encoder = parser.add_argument_group('Encoder')
        encoder.add_argument('--conv_impl', type=str)
        encoder.add_argument('--encode_hidden_units', type=int)
        encoder.add_argument('--dense_spec', type=str)
        _boolean_argument('encode_lstm', encoder)

        network = parser.add_argument_group('Network')
        network.add_argument('--dropout', type=float)
        network.add_argument('--time_steps', type=int)
        _boolean_argument('use_dropout', network)
        _boolean_argument('use_maxnorm', network)
        _boolean_argument('learn_initial_states', network)
        _boolean_argument('use_batch_norm', network)
        _boolean_argument('use_batch_renorm', network)
        network.add_argument('--batch_norm_decay', type=float)
        network.add_argument('-L', '--num_image_patches', type=int)


        decoder = parser.add_argument_group('Decoder')
        decoder.add_argument('--attention_impl', type=str)
        decoder.add_argument('--attention_units', type=int)
        decoder.add_argument('--attention_input', type=str)
        decoder.add_argument('--attention_score_nonlinearity', type=str)
        decoder.add_argument('--num_hidden', type=int)
        decoder.add_argument('--final_activation', type=str)
        decoder.add_argument('--sampling_scheme', type=str)
        decoder.add_argument('--sampling_min', type=float)
        decoder.add_argument('--sampling_decay_steps', type=int)

        registered_params = [action.dest for action in parser._optionals._group_actions[1:]]

        return parser


    def parse_args_and_setup(self):
        defaults = self.defaults
        args = self.parser.parse_args()
        args_dict = vars(args)

        # Convert 0/1 to boolean, keep None
        for argname in self.boolean_args:
            if not args_dict[argname] is None:
                args_dict[argname] = bool(args_dict[argname])

        self.args = args

        # Process while defaults not set yet
        if args.shuffle_examples is None and not args.shuffle_examples_capacity is None:
            raise ValueError('Shuffle capacity given without --shuffle_examples')
        if args.encode_lstm is None and not args.encode_hidden_units is None:
            raise ValueError('Encode hidden units given without --encode_lstm')

        # In order to make it possible to save the absence of a value
        # (args.cson), allow for 'none' and convert to None
        # Kinda silly since a lot of code works with 'none', but oh well
        # none_updates = {}
        # for k, v in args.__dict__.items():
        #     if isinstance(v, str) and v.lower() == 'none':
        #         none_updates[v] = None
        # args.__dict__.update(none_updates)

        defaults.update({ k: v for (k, v) in args.__dict__.items() if v is not None})
        args.__dict__.update(defaults)
        args.debug = self.debug

        if args.gen_log_dir:
            args.log_dir = self._gen_log_dir(args.log_dir, args.prefix)
        else:
            os.makedirs(args.log_dir, exist_ok=True)

        self.setup_log(args.log_dir)
        log = Log.get_logger(__name__)

        if not args.gen_log_dir and args.restore_if_possible is None:
            args.restore_if_possible = True
            log.info('Restoring if possible since a log directory is provided')

        if 'LD_PRELOAD' in os.environ and 'tcmalloc' in os.environ['LD_PRELOAD']:
            log.info('Using tcmalloc, good on you!')
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
        Log.save_hostname()
        Log.save_condor()


    def _gen_log_dir(self, base_dir, prefix='', symlink=True):
        # Create a folder to hold results
        import time
        time_str = time.strftime("%d-%m-%Y-%H-%M-%S", time.gmtime())
        if prefix == '' or prefix is None:
            base_log_dir = time_str
        else:
            base_log_dir = '{}_{}'.format(prefix, time_str)
        log_dir = base_dir + '/' + base_log_dir
        os.makedirs(log_dir, exist_ok=True)

        if symlink:
            link_path = base_dir + '/last'

            try:
                # Clean up old softlink
                os.remove(link_path)
            except OSError as ex:
                pass # All G

            try:
                os.symlink(base_log_dir, link_path)
            except FileExistsError as e:
                # Race conditions, what can you do
                pass

        return log_dir
