#!/usr/bin/env python

import sys
import os
import argparse

from jinja2 import Template

this_dir = os.path.dirname(os.path.realpath(__file__))
tpl_file = os.path.normpath(this_dir + '/job.tpl')

sane_args = dict(
    batch_size=16, val_batch_size=14,
    shuffle_examples=True, shuffle_examples_capacity=64,
    conv_impl=None, attention_impl=None,
    encode_lstm=True, encode_hidden_units=256,
)

def dict_to_args(d):
    arg_list = []

    for k, v in d.items():
        if v is None:
            s = 'none'
        elif isinstance(v, bool):
            s = str(int(v)) # Boolean is represented 0 or 1
        else:
            s = str(v)

        s = '--{}={}'.format(k, s)

        arg_list.append(s)

    return ' '.join(arg_list)



def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--prefer', type=str, default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('-N', '--num_jobs', type=int, default=1, help='# of jobs')
    parser.add_argument('--max_retries', type=int, default=3)
    parser.add_argument('--dataset', type=str, choices=['confer-pts', 'confer-vggface-max'],
                        default='confer-pts')
    args, rest_args = parser.parse_known_args()

    with open(tpl_file, 'r') as f:
        tpl = Template(f.read())

    base_path = '/vol/bitbucket/rv1017'

    pargs = sane_args.copy()
    pargs['log_dir'] = base_path + '/log'
    pargs['data_file'] = base_path + '/data/{}/train.tfrecords'.format(args.dataset)
    pargs['val_data'] = base_path + '/data/{}/val.tfrecords'.format(args.dataset)
    pargs['show_progress_bar'] = False
    if args.prefix and args.prefix != '':
        pargs['prefix'] = args.prefix

    prefix = args.prefix
    if prefix != '':
        prefix = prefix + '_'

    conda_envs = dict(gpu='tf-gpu', cpu='tf-cpu')

    CUDA_ROOT = '/vol/cuda/8.0.61'
    conda_root = '/vol/hmi/projects/ruben/miniconda'
    conda_env = conda_envs[args.prefer]
    env_base = '{}/envs/{}'.format(conda_root, conda_env)

    env = dict(
        # LD_LIBRARY_PATH='/vol/hmi/projects/ruben/miniconda/lib',
        LD_LIBRARY_PATH='{}/lib:$ENV(CUDA_ROOT)/lib:$ENV(LD_LIBRARY_PATH)'.format(env_base),
        PYTHONHOME=env_base
    )

    if args.prefer == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = "-1"

    tpl_args = dict(
        python='{}/bin/python'.format(env_base),
        base='/vol/bitbucket/rv1017',
        env=env,
        prefix=prefix,
        prefer=args.prefer,
        env_string=' '.join('{}={}'.format(k, v) for k, v in env.items()),
        args=dict_to_args(pargs) + ' ' + ' '.join(rest_args),
        N=args.num_jobs,
        max_retries=args.max_retries
    )

    job_desc = tpl.render(**tpl_args)
    print(job_desc)



if __name__ == '__main__':
    main()
