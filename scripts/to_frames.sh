#!/bin/bash

basedir=$(pwd)
baseframedir=${FRAME_DIR:="$basedir/frames"}
baseframedir=$(realpath $baseframedir)
echo "Saving frames to $baseframedir"

for dir in "$@"; do
  if [ -d "$dir" ]; then
    basename=$(basename "$dir")
    framedir="$baseframedir/$basename"
    echo $framedir
    mkdir -p "$framedir"

    cd "$dir"
    vid=$(find -name '*.avi')
    echo $vid

    ffmpeg -i "$vid" "$framedir"/'%04d.jpg'
    cd $basedir
  fi
done

