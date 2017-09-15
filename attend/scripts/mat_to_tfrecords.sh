#!/bin/bash

TMP_DIR=$bb/scratch
TMP_DIR=/home/ruben/tmp
name=$1
matdir=$2

echo "Converting .mat to .hdf5"
./pre/mat_to_hdf5.py $matdir -o $TMP_DIR/$name.hdf5

echo "Converting features with annotations"
h5merge --target $TMP_DIR/$name-annot.hdf5 \
  $TMP_DIR/$name.hdf5 \
  $TMP_DIR/confer-annot.hdf5

echo "Splitting dataset into train/test sets"
./pre/split_dataset.py -i $TMP_DIR/$name-annot.hdf5 -o $TMP_DIR/$name-splits

echo "Converting sets to .tfrecords"
for f in $(find $TMP_DIR/$name-splits -type f); do
  ./pre/hdf5_to_tfrecords.py -i $f
done

