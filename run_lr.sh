#!/bin/bash

set -e

#learning_rates=(1e-3 3e-3 1e-2)
learning_rates=(2e-3 4e-3 5e-3 6e-3 7e-3 8e-3 9e-3)

for max_lr in "${learning_rates[@]}"
do
    # 计算 min_lr = max_lr * 0.1
    min_lr=$(python -c "print(float('$max_lr') * 0.1)")

    echo "======================================"
    echo "Running experiment:"
    echo "max_lr=$max_lr"
    echo "min_lr=$min_lr"
    echo "======================================"

    uv run python training/train.py \
        lr_scheduler.max_learning_rate=$max_lr \
        lr_scheduler.min_learning_rate=$min_lr

done
