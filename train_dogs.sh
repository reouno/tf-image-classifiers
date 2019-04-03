#!/bin/sh
set -eux

python train.py \
    mobilenet_v2_custom_full \
    -d /home/leo/src/dataset/stanford-dogs-dataset/cropped_dataset/train \
    -td /home/leo/src/dataset/stanford-dogs-dataset/cropped_dataset/test \
    -vd /home/leo/src/dataset/stanford-dogs-dataset/cropped_dataset/validation \
    -s tmp/mobilenet_v2_custom_full_1.0_dogs_log_190403 \
    -ts 224 224 \
    -cm sparse \
    -bs 32 \
    -ne 200 \
    -fc 0.4 640 0.2
