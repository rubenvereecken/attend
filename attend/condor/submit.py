#!/usr/bin/env python

import sys
import os
import argparse
import logging
import cson

from jinja2 import Template

this_dir = os.path.dirname(os.path.realpath(__file__))
tpl_file = os.path.normpath(this_dir + '/job.tpl')

sane_args = dict(
    restore_if_possible=True,
    batch_size=16, val_batch_size=14,
    shuffle_examples=True, shuffle_examples_capacity=64,
    conv_impl=None, attention_impl=None,
    encode_lstm=True, encode_hidden_units=256,
)

from attend.util import dict_to_args

def _gen_log_dir(base_dir, prefix='', symlink=True):
    # Create a folder to hold results
    import time
    time_str = time.strftime("%d-%m-%Y-%H-%M-%S", time.gmtime())
    if prefix == '' or prefix is None:
        base_log_dir = time_str
    else:
        base_log_dir = '{}_{}'.format(prefix, time_str)
    log_dir = base_dir + '/' + base_log_dir + '.$(Cluster)'
    return log_dir


def generate_job(prefix, prefer='gpu', rest_args=[], dataset=None, resume_dir=None,
                 base_log_path=None):
    with open(tpl_file, 'r') as f:
        tpl = Template(f.read())

    base_path = '/vol/bitbucket/rv1017'
    if base_log_path is None:
        base_log_path = base_path + '/log'

    # TODO reuse parameters if log directory exists, like `run_like.py`

    if resume_dir:
        with open(resume_dir + '/args.cson', 'r') as f:
            pargs = cson.load(f)
    else:
        pargs = sane_args.copy()
        if prefix and prefix != '':
            pargs['prefix'] = prefix

    pargs['log_dir'] = resume_dir or _gen_log_dir(base_log_path, prefix)
    pargs['gen_log_dir'] = False
    pargs['show_progress_bar'] = False
    prefix = pargs.get('prefix') or prefix
    logging.warning('Submitting with prefix {}'.format(prefix))
    if pargs.get('debug'): pargs.pop('debug')

    data_in_rest_args =any(map(lambda s: 'data_file' in s, rest_args))

    if (not pargs.get('data_file') or dataset) and not data_in_rest_args:
        if not dataset and not data_in_rest_args:
            raise Exception('No dataset specified and no data_files found in args')
            # dataset = 'confer-pts'
        logging.warning('Using dataset {}'.format(dataset))
        pargs['data_file'] = base_path + '/data/{}/train.tfrecords'.format(dataset)
        pargs['val_data'] = base_path + '/data/{}/val.tfrecords'.format(dataset)
    elif data_in_rest_args:
        logging.warning('Relying on data files in rest args')
    else:
        logging.warning('Using data files from resume directory')

    conda_envs = dict(gpu='tf-gpu', cpu='tf-cpu')

    CUDA_ROOT = '/vol/cuda/8.0.61'
    conda_root = '/vol/hmi/projects/ruben/miniconda'
    conda_env = conda_envs[prefer]
    env_base = '{}/envs/{}'.format(conda_root, conda_env)

    env = dict(
        # LD_LIBRARY_PATH='/vol/hmi/projects/ruben/miniconda/lib',
        LD_LIBRARY_PATH='{}/lib:$ENV(CUDA_ROOT)/lib:$ENV(LD_LIBRARY_PATH)'.format(env_base),
        PYTHONHOME=env_base
    )

    if prefer == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = "-1"

    tpl_args = dict(
        python='{}/bin/python'.format(env_base),
        base='/vol/bitbucket/rv1017',
        env=env,
        prefix=prefix,
        prefer=prefer,
        env_string=' '.join('{}={}'.format(k, v) for k, v in env.items()),
        args=dict_to_args(pargs) + ' ' + ' '.join(rest_args)
    )

    job_desc = tpl.render(**tpl_args)
    return job_desc


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--prefer', type=str, default='gpu', choices=['gpu', 'cpu'])
    parser.add_argument('--dataset', type=str,
                        choices=['confer-pts', 'confer-vggface-max',
                                 'confer-expr+sift', 'confer-expr+sift+audio'],
                        default=None)
    parser.add_argument('--resume_dir', type=str, help='To restore from')
    args, rest_args = parser.parse_known_args()

    job = generate_job(args.prefix, args.prefer, rest_args, args.dataset,
                        args.resume_dir)
    print(job)


if __name__ == '__main__':
    main()
