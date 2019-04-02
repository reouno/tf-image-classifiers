#!/bin/sh
set -eux

python train.py \
    fcn \
    -d mnist \
    -s tmp/fcn_mnist_log \
    -ne 10
