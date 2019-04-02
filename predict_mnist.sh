#!/bin/sh
set -eux

python predict.py \
    -t sample/image/mnist_9.jpg \
    -m tmp/fcn_mnist_log/fcn_20190320-070454.h5 \
    -s tmp/fcn_mnist_log \
    -is 28 28 \
    -cm grayscale \
    -nc 10
