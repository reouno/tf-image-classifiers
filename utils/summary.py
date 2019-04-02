import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras
from typing import List, Text, Tuple

def test_summary(save_dir: Text,
                 file_name: Text = 'test_summary.txt',
                 sums: List[Text] = []):



    '''
    This function is deprecated.
    '''



    '''
    :param save_dir: directory where the summary file is saved. we are assuming that the directory is the same as one for saving model/checkpoints.
    :param file_name: summary file name
    :param sums: select summary contents. empty list means summarizing all. to be implemented.
    '''
    if not os.path.isdir(save_dir):
        raise RuntimeError('save_dir is not directory or does not exists, {}\nwe should not create directory inside this function.'.format(save_dir))
    file_path = os.path.join(save_dir, file_name)
    if os.path.exists(file_path):
        raise RuntimeError('file_path already exists, {}'.format(file_path))

    content = 'Test summary\n\n'
    content += 'accuracy: {}\n'.format(test_acc)
    content += 'loss:     {}\n'.format(test_loss)
    with open(file_path, 'w') as f:
        f.write(content)
