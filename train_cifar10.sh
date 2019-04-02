#!/bin/sh
set -eux

python train.py \
    small_cnn \
    -d cifar10 \
    -s tmp/small_cnn_cifar10_log \
    -ne 10
