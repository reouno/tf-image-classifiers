import tensorflow as tf
from tensorflow import keras
from typing import List

def fcn(layers: List[int] = [784,128,10],
        hidden_activation = tf.nn.relu,
        out_activation = tf.nn.softmax):
    '''
    :param layers: list of units of each layer
    :param hidden_activation: activation function of hidden layers
    :param out_activation: activation function of output layers
    '''
    # at leaset input and output layers
    assert len(layers) > 2

    # num units
    in_units = layers[0]
    hidden_layers = layers[1:-1]
    out_units = layers[-1]

    # network definition
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=in_units))
    for num_units in hidden_layers:
        model.add(keras.layers.Dense(num_units, activation=hidden_activation))

    model.add(keras.layers.Dense(out_units, activation=out_activation))
    return model
