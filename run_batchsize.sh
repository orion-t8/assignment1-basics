#!/bin/bash

set -e

batch_sizes=(64)

for batch_size in "${batch_sizes[@]}"
do
    echo "======================================"
    echo "Running experiment:"
    echo "batch_size=$batch_size"
    echo "======================================"

    uv run python training/train.py \
	training.batch_size=$batch_size
done
