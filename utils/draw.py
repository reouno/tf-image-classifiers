import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras
from typing import List, Text, Tuple

def draw_images(images: np.ndarray,
                plt_size: Tuple[int,int] = (5,5),
                fig_size: Tuple[int,int] = (20,20),
                save_dir: Text = '',
                file_name: Text = 'sample_data.png'):
    '''
    :param images: target images
    :param plt_size: (row, col) of subplot. row*col must be equal to the no. of images.
    :param fig_size: (width, height) of figure
    :param save_dir: directory to save figure
    :param file_name: figure file name
    '''
    row = plt_size[0]
    col = plt_size[1]
    assert row * col == images.shape[0]

    if len(save_dir) > 0 and not os.path.isdir(save_dir):
        # if it's empty string, it's assumed that there is no need to save, so skip and proceed the process.
        raise RuntimeError('save_dir is not directory or does not exist, {}'.format(save_dir))
    if len(save_dir) > 0:
        save_path = os.path.join(save_dir, file_name)
        if os.path.exists(save_path):
            raise RuntimeError('save_path already exists, {}'.format(save_path))
    else:
        save_path = None


    img_cnt = 0
    plt.figure(figsize = fig_size)
    plt.ion()
    plt.show()
    for r in range(row):
        for c in range(col):
            plt.subplot(row, col, img_cnt+1)
            plt.imshow(images[img_cnt])
            plt.pause(0.001)
            img_cnt += 1
    if save_path is not None:
        plt.savefig(save_path)
