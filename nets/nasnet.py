from tensorflow import keras
from keras.applications.nasnet import NASNetMobile, NASNetLarge
from keras.layers import Dense, Dropout, Flatten, Input
from keras.models import Model, Sequential
from typing import List, Text, Tuple, Union

def nasnet(
        model_type: Text,
        in_shape: Union[None, Tuple[int, int, int]],
        out_units: int,
        weights: Union[None, Text]=None,
        full_conn: Union[None, List[int]]=None
        ):
    '''NASNet Large (331x331 in default) or Mobile (224x224 in default)
    :param model_type: "Large" or "Mobile"
    :param in_shape: input shape in format (H, W, D), or None
    :param out_units: no. of classes
    :param weights: None or "imagenet". None for random initialization and "imagenet" for using pre-trained weights with imagenet datset.
    :param full_conn: specify last full connection layers structure except output layer. This argument will be used only when `weights` is "imagenet".
    '''
    LARGE = 'LARGE'
    MOBILE = 'MOBILE'

    if weights is None:
        model = nasnet_scratch(model_type, in_shape, out_units)
    elif weights == 'imagenet':
        if full_conn is None:
            raise RuntimeError('invalid arguments, "{}". You must set `full_conn` argument when you set `weights` as "imagenet".'.format(full_conn))
        full_conn = full_conn + [out_units]
        model = nasnet_fine_tune(model_type, in_shape, full_conn)
    else:
        raise RuntimeError('invalid argument. `weights` must be None or "imagenet" but got "{}"'.format(weights))

    return model

def nasnet_scratch(
        model_type: Text,
        in_shape: Union[None, Tuple[int, int, int]],
        out_units: int
        ):
    '''NASNet Large (331x331 in default) or Mobile (224x224 in default)
    :param model_type: "Large" or "Mobile"
    :param in_shape: input shape in format (H, W, D), or None
    :param out_units: no. of classes
    '''
    LARGE = 'LARGE'
    MOBILE = 'MOBILE'

    if model_type.upper() == LARGE:
        model = NASNetLarge(
                input_shape=in_shape,
                include_top=True,
                weights=None,
                input_tensor=None,
                classes=out_units
                )
    elif model_type.upper() == MOBILE:
        model = NASNetMobile(
                input_shape=in_shape,
                include_top=True,
                weights=None,
                input_tensor=None,
                classes=out_units
                )
    else:
        raise RuntimeError('invalid model type argument, "{}". "Large" and "Mobile" are supported.'.format(model_type))

    return model

def nasnet_fine_tune(
        model_type: Text,
        in_shape: Tuple[int, int, int],
        full_conn: List[int]
        ):
    '''NASNet Large (331x331 in default) or Mobile (224x224 in default) for fine-tuning based on pre-trained model with imagenet
    :param model_type: "Large" or "Mobile"
    :param in_shape: input shape in format (H, W, D), or None
    :param full_conn: specify last full connection layers structure except output layer.
    '''
    LARGE = 'LARGE'
    MOBILE = 'MOBILE'

    input_tensor = Input(shape=in_shape)
    if model_type.upper() == LARGE:
        base_model = NASNetLarge(
                include_top=False,
                weights='imagenet',
                input_tensor=input_tensor
                )
    elif model_type.upper() == MOBILE:
        base_model = NASNetMobile(
                include_top=False,
                weights='imagenet',
                input_tensor=input_tensor
                )
    else:
        raise RuntimeError('invalid model type argument, "{}". "Large" and "Mobile" are supported.'.format(model_type))

    # full connection layers to be trained
    if full_conn is None:
        raise RuntimeError('invalid arguments, "{}". You must set `full_conn` argument when you set `weights` as "imagenet".'.format(full_conn))
    else:
        top_model = Sequential()
        top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        for n in full_conn[:-1]: # except the output layer
            assert n > 0
            if n < 1: # means dropout
                top_model.add(Dropout(n))
            else: # full connection layer
                top_model.add(Dense(n, activation='relu'))

        top_model.add(Dense(full_conn[-1], activation='softmax'))
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # freeze convolutional layers
    for layer in base_model.layers:
        layer.trainable = False

    return model
