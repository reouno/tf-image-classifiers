import glob
import os
from tensorflow import keras
from typing import Text, Tuple, Union

import data_generator
from utils.fs import get_all_img_files, list_dir

# name of predefined dataset
DATASET_MNIST = 'MNIST'
DATASET_CIFAR10 = 'CIFAR10'

class Dataset:
    '''load dataset
    '''
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    x_valid = None
    y_valid = None
    train_generator = None
    test_generator = None
    validation_generator = None

    # no. of samples
    num_train = None
    num_test = None
    num_validation = None

    input_shape = None # (H, W, D)
    output_units = None # int

    def __init__(
            self,
            data: Text,
            test_data: Text='',
            validation_data: Text='',
            input_shape: Union[None, Tuple[int, int, int]]=None,
            output_units: int=None,
            color_mode: Text='rgb',
            batch_size: int=32,
            class_mode: Text='categorical',
            ):
        '''
        :param data: dataset name or directory path. If specify detectory path, the sub directory names have to be the class names and the no. of sub directories must be equal to the no. of classes.
        :param input_shape: input shape in the format (H,W,D). This has to be specified if using own dataset, namely `data` is a directory path.
        :param output_units: the number of classes.
        :param test_data: test dataset directory, if any. This argument needs to be set only if `data` is a directory path.
        :param validation_data: validation dataset directory, if any. This argument needs to be set only if `data` is a directory path.
        :param color_mode: color mode of input image data. One of "grayscale", "rgb", and "rgba". This argument needs to be set only if `dataset` is a directory path.
        :param batch_size: batch size for image data generator from directory.
        :param class_mode: classification mode. This argument needs to be set only if `dataset` is a directory path. See "data_generator.py" for more details.
        '''
        self.data = data
        self.test_data = test_data
        self.validation_data = validation_data
        self.input_shape = input_shape
        self.output_units = output_units
        self.color_mode = color_mode
        self.batch_size = batch_size
        self.class_mode = class_mode
        if self.data.upper() == DATASET_MNIST:
            self.__mnist()
        elif self.data.upper() == DATASET_CIFAR10:
            self.__cifar10()
        elif os.path.isdir(self.data):
            self.__data_from_dir()
        else:
            # to be implemented
            raise RuntimeError('invalid dataset name or no such directory, "{}"'.format(data))

    def __mnist(self):
        '''load mnist
        '''
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.mnist.load_data()
        self.input_shape = self.x_train.shape[1:]
        self.output_units = 10
        self.x_train = self.x_train / 255.0 # value range in [0, 1]
        self.x_test = self.x_test / 255.0
        
        # no. of samples
        self.num_train = self.x_train.shape[0]
        self.num_test = self.x_test.shape[0]

    def __cifar10(self):
        '''load cifar10
        '''
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar10.load_data()
        self.input_shape = self.x_train.shape[1:]
        self.output_units = 10
        self.x_train = self.x_train / 255.0 # value range in [0, 1]
        self.x_test = self.x_test / 255.0
        
        # no. of samples
        self.num_train = self.x_train.shape[0]
        self.num_test = self.x_test.shape[0]

    def __data_from_dir(self):
        '''create image data generator from directory
        '''
        self.output_units = len(list_dir(self.data))

        # training data
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True
                )
        self.train_generator = train_datagen.flow_from_directory(
                self.data,
                target_size=self.input_shape[:2],
                color_mode=self.color_mode,
                batch_size=self.batch_size,
                class_mode=self.class_mode
                )

        # test data
        test_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255
                )
        self.test_generator = test_datagen.flow_from_directory(
                self.test_data,
                target_size=self.input_shape[:2],
                color_mode=self.color_mode,
                batch_size=self.batch_size,
                class_mode=self.class_mode
                )

        # validation data
        validation_datagen = keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255
                )
        self.validation_generator = validation_datagen.flow_from_directory(
                self.validation_data,
                target_size=self.input_shape[:2],
                color_mode=self.color_mode,
                batch_size=self.batch_size,
                class_mode=self.class_mode
                )

        # no. of samples
        self.num_train = len(get_all_img_files(self.data))
        self.num_test = len(get_all_img_files(self.test_data))
        self.num_validation = len(get_all_img_files(self.validation_data))


if __name__ == '__main__':
    dataset = Dataset('mnist')
    assert dataset.input_shape == (28, 28)
    assert dataset.output_units == 10
    assert dataset.num_train == 60000
    assert dataset.num_test == 10000
    assert dataset.num_validation is None
    print('OK')
