#!/bin/bash

n_folds=5
basedir=$(pwd)
infile=$1
outdir=$2
dset=$3
dset=${dset:-two}
folddir="$basedir/data/confer/protocol/$dset"

for (( i=1; i<=$n_folds; i++ ))
do
  foldfile="$folddir/FOLD_$i.txt"
  python pre/split_dataset.py $infile -k $foldfile -o "$outdir/$i.hdf5"
done
