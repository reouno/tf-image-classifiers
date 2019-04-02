import tensorflow as tf
from tensorflow import keras
from typing import Text, Tuple

def img_data_gen(x_train, y_train, x_validation, y_validation, batch_size: int):
    '''
    :param x_train: image dataset in keras dataset format
    :param y_train: target data in keras dataset format
    :param x_validation: image dataset in keras dataset format
    :param y_validation: target data in keras dataset format
    :param batch_size: batch size
    '''
    # args
    data_gen_args = dict(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
            )

    # generator
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed=1
    augment=True
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
            **data_gen_args
            )
    validation_datagen = keras.preprocessing.image.ImageDataGenerator(
            **data_gen_args
            )
    train_datagen.fit(x_train, augment=augment, seed=seed)
    validation_datagen.fit(x_validation, augment=augment, seed=seed)
    train_generator = train_datagen.flow(
            x_train,
            y_train,
            batch_size=batch_size
            )
    validation_generator = train_datagen.flow(
            x_train,
            y_train,
            batch_size=batch_size
            )
    return train_generator, validation_generator

def img_data_gen_from_dir(train_dir: Text,
                          test_dir: Text,
                          validation_dir: Text,
                          target_size: Tuple[int, int],
                          batch_size: int,
                          class_mode: Text):
    '''
    :param train_dir: training dataset directory
    :param test_dir: test dataset directory
    :param validation_dir: validataion dataset directory
    :param target_size: target image size (height, width)
    :param batch_size: batch size
    :param class_mode: class mode. "categorical", "binary", "sparse", "input", or "None"
    '''

    COLOR_MODE = 'rgb' # one of 'grayscale', 'rgb', 'rgba'

    # training data
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
            )
    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            color_mode=COLOR_MODE,
            batch_size=batch_size,
            class_mode=class_mode
            )

    # test data
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
            )
    test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=target_size,
            color_mode=COLOR_MODE,
            batch_size=batch_size,
            class_mode=class_mode
            )

    # validation data
    validation_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
            )
    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=target_size,
            color_mode=COLOR_MODE,
            batch_size=batch_size,
            class_mode=class_mode
            )

    return train_generator, test_generator, validation_generator
