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
    parser = argparse.ArgumentParser(description='Convert frame images to hf5 and preprocess')
    parser.add_argument('--prefix', type=str, default='')
    args, rest_args = parser.parse_known_args()

    with open(tpl_file, 'r') as f:
        tpl = Template(f.read())

    base_path = '/vol/bitbucket/rv1017'

    pargs = sane_args.copy()
    pargs['log_dir'] = base_path + '/log'
    pargs['data_file'] = base_path + '/confer-splits/train.tfrecords'
    pargs['val_data'] = base_path + '/confer-splits/val.tfrecords'
    pargs['show_progress_bar'] = False
    pargs['prefix'] = args.prefix

    prefix = args.prefix
    if prefix != '':
        prefix = prefix + '_'

    CUDA_ROOT = '/vol/cuda/8.0.61'

    env = dict(
        # LD_LIBRARY_PATH='/vol/hmi/projects/ruben/miniconda/lib',
        LD_LIBRARY_PATH='/vol/hmi/projects/ruben/miniconda/lib:$ENV(CUDA_ROOT)/lib:$ENV(LD_LIBRARY_PATH)',
        PYTHONHOME='/vol/hmi/projects/ruben/miniconda'
    )

    tpl_args = dict(
        python='/vol/hmi/projects/ruben/miniconda/bin/python',
        base='/vol/bitbucket/rv1017',
        env=env,
        prefix=prefix,
        env_string=' '.join('{}={}'.format(k, v) for k, v in env.items()),
        args=dict_to_args(pargs) + ' ' + ' '.join(rest_args)
    )

    job_desc = tpl.render(**tpl_args)
    print(job_desc)



if __name__ == '__main__':
    main()
