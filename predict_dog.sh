#!/bin/sh

IMG_FILE=$1

python predict.py \
    -t ${IMG_FILE} \
    -m tmp/mobilenet_v2_0.7_dogs_log_190320/mobilenet_v2_20190320-142041.h5 \
    -s tmp/mobilenet_v2_0.7_dogs_log_190320 \
    -is 224 224 \
    -cm rgb \
    -nc 120
