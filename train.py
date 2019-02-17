#!/usr/bin/env python

import datetime
import matplotlib.pyplot as plt
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from typing import Text

import call_backs as cb
from get_args import arg_parser
from nets.fcn import fcn
import utils

def main(network: Text,
          dataset: Text,
          save_dir: Text,
          batch_size: int=32,
          num_epochs: int=1):
    '''
    :param network: name of network
    :param dataset: dataset directory or dataset name
    :param save_dir: directory where the models/checkpoints are saved
    :param batch_size: batch size
    :param num_epochs: the number of epochs
    '''

    # create directory for saving weights and model
    assert len(save_dir) > 0
    if os.path.exists(save_dir):
        while True:
            choice = input('Delete existing "{}"?[y/N]: '.format(save_dir))
            if choice.upper() in ['Y','YES']:
                shutil.rmtree(save_dir)
                break
            elif choice.upper() in ['N','NO']:
                print('Delete/move "{}", or specify another path'.format(save_dir))
                exit()
    try:
        os.makedirs(save_dir)
    except OSError as e:
        print("Cannot create direcotry to save weights and model.\nError message:",e)
        print('Use "-s" option to set directory path.')
        exit()

    # load data
    if dataset.upper() == 'MNIST':
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    else:
        # to be implemented
        raise RuntimeError('no such dataset, {}'.format(dataset))

    # show sample
    if dataset.upper() == 'MNIST':
        utils.draw_images(x_train[:25], plt_size=(5,5), save_dir=save_dir)

    # preprocessing
    if dataset.upper() == 'MNIST':
        x_train = x_train / 255.0 # [0, 1]
        x_test = x_test / 255.0
        print('x_train.shape:',x_train.shape)
        assert len(x_train.shape) == 3
    else:
        # to be implemented
        raise RuntimeError('no such dataset, {}'.format(dataset))

    # load network
    if network.upper() == 'FCN':
        model = fcn([x_train.shape[1:],128,10])
    else:
        # to be implemented
        raise RuntimeError('no such network name, {}'.format(network))

    # compile model
    model.compile(optimizer=tf.train.AdamOptimizer(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

    # model summary
    model_summary_lines = ['Model Summary']
    model.summary(print_fn = model_summary_lines.append)
    model_summary = '\n'.join(model_summary_lines)
    print(model_summary)
    model_sum_file = 'model_summary.txt'
    model_sum_path = os.path.join(save_dir, model_sum_file)
    if os.path.exists(model_sum_path):
        raise RuntimeError('model_sum_path already exists, {}'.format(model_sum_path))
    with open(model_sum_path, 'w') as f:
        f.write(model_summary)

    # set callbacks
    batch_stats = cb.CollectBatchStats()
    checkpoint = cb.modelCheckpoint(save_dir)
    tensorboard = cb.tensorboard(save_dir)

    print('x_train.shape:',x_train.shape)
    print('y_train.shape:',y_train.shape)
    # train
    model.fit(x_train,
              y_train,
              batch_size=batch_size,
              epochs=num_epochs,
              callbacks=[checkpoint, batch_stats, tensorboard])

    # save the model
    dt = '{0:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())
    filename = '{}_{}.h5'.format(network, dt)
    filepath = os.path.join(save_dir, filename)
    model.save(filepath)

    # stats
    plt.figure()
    plt.ylabel('Loss')
    plt.xlabel('Training Steps')
    plt.ylim([0,2])
    plt.plot(batch_stats.batch_losses)
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))

    plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('Training Steps')
    plt.ylim([0,1])
    plt.plot(batch_stats.batch_acc)
    plt.savefig(os.path.join(save_dir, 'train_acc.png'))

    # test & test summary
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    content = 'Test summary\n\n'
    content += 'accuracy: {}\n'.format(test_acc)
    content += 'loss:     {}\n'.format(test_loss)
    file_name = 'test_summary.txt'
    file_path = os.path.join(save_dir, file_name)
    if os.path.exists(file_path):
        raise RuntimeError('file_path already exists, {}'.format(file_path))
    with open(file_path, 'w') as f:
        f.write(content)


if __name__ == '__main__':

    # get args
    parser = arg_parser()
    a = parser.parse_args()

    main(a.net,
         a.dataset,
         a.save_dir,
         batch_size=a.batch_size,
         num_epochs=a.num_epochs)
