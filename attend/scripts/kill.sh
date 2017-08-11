#!/bin/bash

log_dir=$1
pid=$(cat $log_dir/pid)
host=$(cat $log_dir/host)

echo "Killing pid $pid @ $host"

if [ "$host" == "$(hostname)" ]
then
  kill $pid
else
  ssh rv1017@$host kill $pid
fi
