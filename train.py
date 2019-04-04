#!/usr/bin/env pythonelif os.path

import csv
import datetime
import math
import matplotlib.pyplot as plt
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from typing import List, Text, Tuple, Union

import call_backs as cb
import data_generator as dg
from dataset import Dataset
from get_args import arg_parser
#from nets.fcn import fcn
from logger import Logger
import nets.fcn
import nets.cnn_basic
import nets.mobilenet_v2
import nets.nasnet
import nets.xception
import utils.draw
import utils.fs

DATASET_MNIST = 'MNIST'
DATASET_CIFAR10 = 'CIFAR10'

NET_FCN = 'FCN'
NET_SML_CNN = 'SMALL_CNN'
NET_MID_CNN = 'MIDDLE_CNN'
NET_MID_CNN_DP = 'MIDDLE_CNN_DP'
NET_MBN = 'MOBILENET'
NET_MBNV2 = 'MOBILENET_V2'
NET_MBNV2_CUSTOM_FULL = 'MOBILENET_V2_CUSTOM_FULL'
NET_NAS_L = 'NASNET_LARGE'
NET_NAS_M = 'NASNET_MOBILE'
NET_XCEPTION = 'XCEPTION'

# for Mobilenet
ALPHA_FOR_MBN = 1.0
DROPOUT_FOR_MBN = 1e-3

# for Mobilenet V2
ALPHA_FOR_MBN2 = 1.0

# for Mobilenet and Mobilenet V2
ORIGINAL = 'ORIGINAL'
CUSTOM_FULL = 'CUSTOM_FULL'

# for NASNet
NAS_LARGE = 'LARGE'
NAS_MOBILE = 'MOBILE'

BATCH_SIZE_FOR_DATA_GEN = 32
VALIDATION_STEPS = 100

#STEPS_PER_EPOCH_FOR_DATA_FROM_DIR = 500
num_train_data = 15480

def main(
        network: Text,
        dataset: Text,
        save_dir: Text,
        batch_size: int=32,
        num_epochs: int=1,
        target_size: Tuple[int, int]=(224,224),
        class_mode: Text='categorical',
        test_data: Text='',
        validation_data: Text='',

        weights: Union[None, Text]=None,
        full_conn: Union[None, List[int]]=None,

        training: bool=True,
        test: bool=True
        ):
    '''
    :param network: name of network
    :param dataset: dataset directory or dataset name
    :param save_dir: directory where the models/checkpoints are saved
    :param batch_size: batch size
    :param num_epochs: the number of epochs
    :param target_size: target image size. This argument needs to be set only if `dataset` is a directory path.
    :param class_mode: classification mode. This argument needs to be set only if `dataset` is a directory path. See "data_generator.py" for more details.
    :param validation_data: validation dataset directory, if any. This argument needs to be set only if `dataset` is a directory path.
    :param test_data: test dataset directory, if any. This argument needs to be set only if `dataset` is a directory path.
    :param weights: None or "imagenet". None for random initialization and "imagenet" for using pre-trained weights with imagenet datset.
    :param full_conn: specify last full connection layers structure except output layer. This argument will be used only when `weights` is "imagenet".
    :param training: execute trainig or not. Default value is True.
    :param training: execute test or not. Default value is True.
    '''

    # logger
    log = Logger(level='debug')

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

    # validate directory
    if len(validation_data) > 0 and not os.path.isdir(validation_data):
        raise RuntimeError('no such direcotry, "{}".'.format(validation_data))
    if len(test_data) > 0 and not os.path.isdir(test_data):
        raise RuntimeError('no such direcotry, "{}".'.format(test_data))

    # load data
    if dataset.upper() == DATASET_MNIST:
        #(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        data = Dataset('mnist')
        assert data.input_shape == (28, 28)
        assert data.output_units == 10
        assert data.num_train == 60000
        assert data.num_test == 10000
        assert data.num_validation is None
    elif dataset.upper() == DATASET_CIFAR10:
        #(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        data = Dataset('cifar10')
        assert data.input_shape == (32, 32, 1)
        assert data.output_units == 10
        assert data.num_train == 50000
        assert data.num_test == 10000
        assert data.num_validation is None
    elif os.path.isdir(dataset):
        data = Dataset(
                dataset,
                test_data=test_data,
                validation_data=validation_data,
                input_shape=target_size + (3,),
                batch_size=batch_size,
                class_mode=class_mode
                )
        assert data.train_generator is not None
        assert data.test_generator is not None
        assert data.validation_generator is not None
        assert type(data.num_train) is int
        assert type(data.num_test) is int
        assert type(data.num_validation) is int
        print(data.num_train)
        print(data.num_test)
        print(data.num_validation)
    else:
        # to be implemented
        # or invalid dataset name or directory name
        raise RuntimeError('no such dataset or directory, "{}".'.format(dataset))

    # load network
    if (weights is not None) and ('.h5' in weights): # not only weights but entire model
        model = keras.models.load_model(weights)
        # compile model
        model.compile(optimizer=tf.train.AdamOptimizer(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    else:
        if network.upper() == NET_FCN:
            model = nets.fcn.fcn([data.input_shape,128,data.output_units])
        elif network.upper() == NET_SML_CNN:
            model = nets.cnn_basic.cnn('small_cnn', data.input_shape, data.output_units)
        elif network.upper() == NET_MID_CNN:
            model = nets.cnn_basic.cnn('middle_cnn', data.input_shape, data.output_units)
        elif network.upper() == NET_MID_CNN_DP:
            model = nets.cnn_basic.cnn('middle_cnn_dp', data.input_shape, data.output_units)
        elif network.upper() == NET_MBN:
            model = nets.mobilenet.mobilenet(
                    ORIGINAL,
                    data.input_shape,
                    data.output_units,
                    dropout=DROPOUT_FOR_MBN,
                    alpha=ALPHA_FOR_MBN
                    )
        elif network.upper() == NET_MBNV2:
            model = nets.mobilenet_v2.mobilenet_v2(
                    ORIGINAL,
                    data.input_shape,
                    data.output_units,
                    alpha=ALPHA_FOR_MBN2
                    )
        elif network.upper() == NET_MBNV2_CUSTOM_FULL:
            model = nets.mobilenet_v2.mobilenet_v2(
                    CUSTOM_FULL,
                    data.input_shape,
                    data.output_units,
                    weights=weights,
                    full_conn=full_conn,
                    alpha=ALPHA_FOR_MBN2
                    )
        elif network.upper() == NET_NAS_L:
            model = nets.nasnet.nasnet(
                    NAS_LARGE,
                    data.input_shape,
                    data.output_units,
                    weights=weights,
                    full_conn=full_conn
                    )
        elif network.upper() == NET_NAS_M:
            model = nets.nasnet.nasnet(
                    NAS_MOBILE,
                    data.input_shape,
                    data.output_units
                    )
        elif network.upper() == NET_XCEPTION:
            model = nets.xception.xception(
                    data.input_shape,
                    data.output_units,
                    weights=weights,
                    full_conn=full_conn
                    )
        else:
            # to be implemented
            raise RuntimeError('no such network name, {}'.format(network))

        # compile model
        model.compile(optimizer=tf.train.AdamOptimizer(),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        # load weights
        # you can set just ckpt file path if "*.ckpt" file exists.
        # you can set file path like "*.ckpt" same as the above even if "*.ckpt" file
        # doesn't exists but there are files like "*.ckpt.index",
        # "*.ckpt.meta", and "*.ckpt.data-00000-of-00001".
        if (weights is not None) and ('.ckpt' in weights):
            model.load_weights(weights)

    for i,layer in enumerate(model.layers):
        print(i, layer.name)

    # model summary
    model_summary_lines = ['Model Summary']
    model.summary(print_fn = model_summary_lines.append)
    model_summary = '\n'.join(model_summary_lines)
    log.info(model_summary)
    model_sum_file = 'model_summary.txt'
    model_sum_path = os.path.join(save_dir, model_sum_file)
    if os.path.exists(model_sum_path):
        raise RuntimeError('model_sum_path already exists, {}'.format(model_sum_path))
    with open(model_sum_path, 'w') as f:
        f.write(model_summary)

    if training:
        # set callbacks
        batch_stats = cb.CollectBatchStats()
        checkpoint = cb.modelCheckpoint(save_dir)
        tensorboard = cb.tensorboard(save_dir)
    
        # train
        if dataset.upper() in [DATASET_MNIST, DATASET_CIFAR10]:
            history = model.fit(data.x_train,
                      data.y_train,
                      batch_size=batch_size,
                      epochs=num_epochs,
                      callbacks=[checkpoint, batch_stats, tensorboard]
                      )
        elif os.path.isdir(dataset):
            history = model.fit_generator(
                    data.train_generator,
                    steps_per_epoch=math.ceil(data.num_train / batch_size),
                    epochs = num_epochs,
                    validation_data=data.validation_generator,
                    validation_steps=VALIDATION_STEPS,
                    callbacks=[checkpoint, batch_stats, tensorboard]
                    )

        # stats
        # training loss every batch
        plt.figure()
        plt.ylabel('Loss')
        plt.xlabel('Training Steps')
        plt.ylim([0,2])
        plt.plot(batch_stats.batch_losses)
        plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    
        # training accuracy every batch
        plt.figure()
        plt.ylabel('Accuracy')
        plt.xlabel('Training Steps')
        plt.ylim([0,1])
        plt.plot(batch_stats.batch_acc)
        plt.savefig(os.path.join(save_dir, 'train_acc.png'))

        # train loss and val loss every epoch
        plt.figure()
        plt.ylabel('Loss')
        plt.xlabel('Training Epochs')
        #plt.ylim([0,2])
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.legend(bbox_to_anchor=(1,1))
        plt.savefig(os.path.join(save_dir, 'train_loss_epoch.png'))

        # train loss and val loss every epoch
        plt.figure()
        plt.ylabel('Accuracy')
        plt.xlabel('Training Epochs')
        plt.ylim([0,1])
        plt.plot(history.history['acc'], label='acc')
        plt.plot(history.history['val_acc'], label='val_acc')
        plt.legend(bbox_to_anchor=(0.3,1))
        plt.savefig(os.path.join(save_dir, 'train_acc_epoch.png'))

    # save the model
    dt = '{0:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())
    filename = '{}_{}.h5'.format(network, dt)
    filepath = os.path.join(save_dir, filename)
    model.save(filepath)

    # save the class labels
    if os.path.isdir(dataset):
        class_indices = sorted([[v,k] for k,v in data.train_generator.class_indices.items()], key=lambda x:x[0])
        with open(os.path.join(save_dir, 'class_indices.csv'), 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerows(class_indices)

    if test:
        # test & test summary
        if dataset.upper() in [DATASET_MNIST, DATASET_CIFAR10]:
            test_loss, test_acc = model.evaluate(data.x_test, data.y_test)
        elif os.path.isdir(dataset):
            test_loss, test_acc = model.evaluate_generator(data.test_generator, steps=data.num_test)
        #test_loss, test_acc = model.evaluate(x_test, y_test)
        log.info('Test accuracy: {}'.format(test_acc))
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

    weights = None if len(a.weights)==0 else a.weights

    # full_conn parser
    def to_int(x):
        xi = int(x)
        if x == xi:
            return xi
        else:
            return x
    full_conn = list(map(to_int, a.full_conn))
    full_conn = None if len(full_conn)==0 else full_conn

    main(a.net,
         a.dataset,
         a.save_dir,
         batch_size=a.batch_size,
         num_epochs=a.num_epochs,
         target_size=tuple(a.target_size),
         class_mode=a.class_mode,
         test_data=a.test,
         validation_data=a.validation,
         
         weights=weights,
         full_conn=full_conn,

         training=not a.no_training,
         test=not a.no_test
         )
