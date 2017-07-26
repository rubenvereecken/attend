#!/bin/bash

  # -i /vol/bitbucket/rv1017/confer-splits/train.tfrecords \
  # --val_data /vol/bitbucket/rv1017/confer-splits/val.tfrecords \
./train.py \
 -i /var/tmp/data/confer-splits/train.tfrecords \
 --val_data=/var/tmp/data/confer-splits/val.tfrecords \
  --batch_size=16 --val_batch_size=14 \
  --shuffle_examples --shuffle_examples_capacity=64 \
  --conv_impl=none \
  --attention_impl=none \
  --encode_lstm --encode_hidden_units=256 \
  --log_dir=$bb/log $@
