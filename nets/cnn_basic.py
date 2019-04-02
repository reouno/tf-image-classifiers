import tensorflow as tf
from tensorflow import keras


def cnn(name, in_shape, out_units):
    '''
    :param name: name of pre defined cnn
    :param in_shape: input shape (height, width, channel)
    :param out_units: number of units in output layer
    '''
    if name.upper() == 'SMALL_CNN':
        return small_cnn(in_shape, out_units)
    elif name.upper() == 'MIDDLE_CNN':
        return middle_cnn(in_shape, out_units)
    elif name.upper() == 'MIDDLE_CNN_DP':
        return middle_cnn_dp(in_shape, out_units, drop_out=0.3)
    else:
        # to be implemented
        raise RuntimeError('no such name of model, {}'.format(name))

def small_cnn(in_shape, out_units):
    '''
    :param in_shape: input shape (height, width, channel)
    :param out_units: number of units in output layer

    input(H, W, D)
    -> conv(64)
    -> pool
    -> conv(128)
    -> pool
    -> dense(1024)
    -> dense(512)
    -> output(N)
    '''
    model = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=in_shape),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(out_units, activation='softmax')
            ])
    return model

def middle_cnn(in_shape, out_units):
    '''
    :param in_shape: input shape (height, width, channel)
    :param out_units: number of units in output layer

    input(H, W, D)
    -> conv(64)
    -> pool
    -> conv(128)
    -> pool
    -> conv(256)
    -> conv(256)
    -> pool
    -> dense(1024)
    -> dense(512)
    -> output(N)
    '''
    model = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=in_shape),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'),
            keras.layers.Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dense(out_units, activation='softmax')
            ])
    return model

def middle_cnn_dp(in_shape, out_units, drop_out=0.3):
    '''
    :param in_shape: input shape (height, width, channel)
    :param out_units: number of units in output layer
    :param drop_out: a fraction rate of units to 0 at each update during trainig time

    :param
    input(H, W, D)
    -> conv(64)
    -> pool
    -> dp
    -> conv(128)
    -> pool
    -> dp
    -> conv(256)
    -> conv(256)
    -> pool
    -> dp
    -> dense(1024)
    -> dp
    -> dense(512)
    -> dp
    -> output(N)
    '''
    model = keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=in_shape),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Dropout(drop_out),
            keras.layers.Conv2D(filters=128, kernel_size=2, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Dropout(drop_out),
            keras.layers.Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'),
            keras.layers.Conv2D(filters=256, kernel_size=2, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(pool_size=2),
            keras.layers.Dropout(drop_out),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dropout(drop_out),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dropout(drop_out),
            keras.layers.Dense(out_units, activation='softmax')
            ])
    return model

