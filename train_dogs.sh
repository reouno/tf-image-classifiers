#!/bin/sh
set -eux

python train.py \
    xception \
    -d /home/leo/src/dataset/stanford-dogs-dataset/cropped_dataset/train \
    -td /home/leo/src/dataset/stanford-dogs-dataset/cropped_dataset/test \
    -vd /home/leo/src/dataset/stanford-dogs-dataset/cropped_dataset/validation \
    -s tmp/xception_fine_tuning_dogs_log_190402 \
    -ts 299 299 \
    -cm sparse \
    -bs 32 \
    -ne 200 \
    -wt imagenet \
    -fc 2048 0.2 1024 0.2
