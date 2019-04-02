from tensorflow import keras
from keras.applications.xception import Xception
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential
from typing import List, Text, Tuple, Union

def xception(
        in_shape: Union[None, Tuple[int, int, int]],
        out_units: int,
        weights: Union[None, Text]=None,
        full_conn: Union[None, List[int]]=None
        ):
    '''Xception network (299x299 in default)
    :param in_shape: input shape in format (H, W, D), or None
    :param out_units: no. of classes
    :param weights: None or "imagenet". None for random initialization and "imagenet" for using pre-trained weights with imagenet datset.
    :param full_conn: specify last full connection layers structure except output layer. This argument will be used only when `weights` is "imagenet".
    '''
    if weights is None:
        'for training from scratch'
        model = Xception(
                include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=out_units
                )
    else:
        input_tensor = Input(shape=in_shape)
        base_model = Xception(
                include_top=False,
                weights=weights,
                input_tensor=input_tensor
                )

        if full_conn is None:
            raise RuntimeError('invalid arguments, "{}". You must set `full_conn` argument when you set `weights` as "imagenet".'.format(full_conn))
        else:
            top_model = Sequential()
            top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
            for n in full_conn:
                assert n > 0
                if n < 1: # means dropout
                    top_model.add(Dropout(n))
                else: # full connection layer
                    top_model.add(Dense(n, activation='relu'))

            top_model.add(Dense(out_units, activation='softmax'))
            model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

        # freeze convolutional layers
        for layer in base_model.layers:
            layer.trainable = False



    return model
