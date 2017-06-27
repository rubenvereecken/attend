#!/bin/bash

gpu_idxs=$(nvidia-smi --query-gpu=index --format=csv,noheader)
free_gpus=()

for i in $gpu_idxs; do
  mem_used=$(nvidia-smi -i $i -q -d MEMORY | grep Used | head -n1 | awk '{print $3}')
  if [ $mem_used -eq 0 ]; then
    echo $i
    free_gpus+=$i
  fi
done

# echo $free_gpus
