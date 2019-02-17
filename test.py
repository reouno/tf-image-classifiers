#!/usr/bin/env python

'''
This module is deprecated.
'''

import argparse
import datetime
import matplotlib.pyplot as plt
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from typing import Text

from get_args import arg_parser
from nets.fcn import fcn

def main(network: Text,
         dataset: Text,
         save_dir: Text):
    '''
    :param network: name of network
    :param dataset: dataset directory or dataset name
    :param save_dir: directory where the models/checkpoints are saved
    '''

    # check if save_dir exists
    # it should be created when training.
    if not os.path.isdir(save_dir):
        print('save_dir is not directory or does not exists, {}'.format(save_dir))
        print('Train model first')
        exit()

    # read data

    # test

    # summary
