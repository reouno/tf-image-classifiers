from tensorflow import keras
from keras.applications import MobileNet
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, Input
from keras.models import Model, Sequential
from typing import List, Text, Tuple, Union

def mobilenet(
        model_type: Text,
        in_shape: Union[None, Tuple[int, int, int]],
        out_units: int,
        weights: Union[None, Text]=None,
        full_conn: Union[None, List[int]]=None,
        alpha: float=1.0
        ):
    '''mobilenet
    :param model_type: "original" or "custom_full"
    :param in_shape: input shape in format (H, W, D), or None
    :param out_units: no. of classes
    :param weights: None or "imagenet". None for random initialization and "imagenet" for using pre-trained weights with imagenet datset.
    :param full_conn: specify last full connection layers structure except output layer. This argument will be used only when `weights` is "imagenet".
    :param alpha: alpha value to adjust the no. of filters
    '''
    ORIGINAL = 'ORIGINAL'
    CUSTOM_FULL = 'CUSTOM_FULL'

    if model_type.upper() == ORIGINAL:
        model = mobilenet_org(in_shape, out_units, alpha=alpha)
    elif model_type.upper() == CUSTOM_FULL:
        if full_conn is None:
            full_conn = []
        model = mobilenet_custom_full_conn(
                in_shape,
                weights,
                full_conn + [out_units],
                alpha=alpha
                )
    else:
        # to be implemented
        raise RuntimeError('invalid model type argument, "{}". "ORIGINAL" and "CUSTOM_FULL" are supported.'.format(model_type))

    return model


def mobilenet_org(
        in_shape,
        out_units,
        dropout=1e-3,
        alpha=1.0
        ):
    '''mobilenet
    :param in_shape: input shape (height, width, channel)
    :param out_units: number of units in output layer
    :param alpha: alpha value to adjust the no. of filters
    '''

    model = MobileNet(
            input_shape=in_shape,
            alpha=alpha,
            depth_multiplier=1,
            dropout=dropout,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None,
            classes=out_units
            )

    return model


