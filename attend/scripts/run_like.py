#!/usr/bin/env python

import os
import sys
import argparse
import shutil

import cson

def main():
    parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
    parser.add_argument('log_dir', type=str)
    parser.add_argument('-e', '--executable', type=str, default=None)
    parser.add_argument('-n', '--dry-run', action='store_true')
    parser.add_argument('-c', '--confirm', action='store_true')
    parser.add_argument('--load_env', dest='load_env', action='store_true')
    parser.set_defaults(dry_run=False, confirm=False)
    args, rest_args = parser.parse_known_args()

    if args.executable is None:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        args.executable = os.path.normpath(this_dir + '/../train.py')

    if not os.path.exists(args.executable):
        raise Exception('Could not find executable {}'.format(args.executable))

    if not os.path.exists(args.log_dir):
        raise Exception('Directory does not exist')

    # from attend import Log
    # pargs = Log.get_args(args.log_dir)
    with open(args.log_dir + '/args.cson', 'r') as f:
        pargs = cson.load(f)

    if args.load_env:
        with open(args.log_dir + '/env.cson', 'r') as f:
            env = cson.load(f)
    else:
        env = os.environ.copy()

    # $log_dir/$prefix-$timestamp
    pargs.pop('debug')
    old_log_dir = pargs.pop('log_dir') # Generate a new one anyway
    log_dir = '/'.join(old_log_dir.split('/')[:-1])
    pargs['log_dir'] = log_dir
    pargs['gen_log_dir'] = True

    arg_list = []

    for k, v in pargs.items():
        if isinstance(v, str):
            s = "{}".format(v)
        elif isinstance(v, bool):
            s = str(int(v)) # Boolean is represented 0 or 1
        else:
            s = str(v)

        s = '--{}={}'.format(k, s)

        arg_list.append(s)

    arg_string = ' '.join(arg_list)
    rest_string = ' '.join(rest_args)
    full_string = '{} {} {}'.format(args.executable, arg_string, rest_string)

    if args.dry_run:
        print(full_string)
        sys.exit(0)

    if args.confirm:
        print(full_string)
        y = input('Run? (Y/n) ')
        if y.strip() != 'y' and y.strip() != '':
            sys.exit(1)

    import subprocess
    env.update(dict(PATH=os.environ['PATH']))

    result = subprocess.call([args.executable, *arg_list, *rest_args], env=env)
    print(result)


if __name__ == '__main__':
    main()
