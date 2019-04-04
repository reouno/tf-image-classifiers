from tensorflow import keras
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, Input
from keras.models import Model, Sequential
from typing import List, Text, Tuple, Union

def mobilenet_v2(
        model_type: Text,
        in_shape: Union[None, Tuple[int, int, int]],
        out_units: int,
        weights: Union[None, Text]=None,
        full_conn: Union[None, List[int]]=None,
        alpha: float=1.0
        ):
    '''mobilenet v2
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
        model = mobilenet_v2_org(in_shape, out_units, alpha=alpha)
    elif model_type.upper() == CUSTOM_FULL:
        if full_conn is None:
            full_conn = []
        model = mobilenet_v2_custom_full_conn2(
                in_shape,
                weights,
                full_conn + [out_units],
                alpha=alpha
                )
    else:
        # to be implemented
        raise RuntimeError('invalid model type argument, "{}". "ORIGINAL" and "CUSTOM_FULL" are supported.'.format(model_type))

    return model


def mobilenet_v2_org(
        in_shape,
        out_units,
        alpha=1.0
        ):
    '''mobilenet v2
    :param in_shape: input shape (height, width, channel)
    :param out_units: number of units in output layer
    :param alpha: alpha value to adjust the no. of filters
    '''

    model = MobileNetV2(
            input_shape=in_shape,
            alpha=alpha,
            depth_multiplier=1,
            include_top=True,
            weights=None,
            input_tensor=None,
            pooling=None,
            classes=out_units
            )

    return model

def mobilenet_v2_custom_full_conn(
        in_shape: Tuple[int, int, int],
        weights: Union[None, Text],
        full_conn: List[int],
        alpha: float=1.0
        ):
    '''mobilenet v2 with customized full connection layers
    :param in_shape: input shape (height, width, channel)
    :param full_conn: specify last full connection layers structure except output layer.
    :param alpha: alpha value to adjust the no. of filters
    '''
    input_tensor = Input(shape=in_shape)
    base_model = MobileNetV2(
            include_top=False,
            weights=weights,
            input_tensor=input_tensor
            )

    # full connection layers
    if full_conn is None:
        raise RuntimeError('invalid arguments, "{}". You must set `full_conn` argument for this network.'.format(full_conn))
    else:
        top_model = Sequential()
        #top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
        top_model.add(GlobalAveragePooling2D(name='custom_global_average_pooling_2d'))
        for i, n in enumerate(full_conn[:-1]): # except the output layer
            assert n > 0
            if n < 1: # means dropout
                top_model.add(Dropout(n, name='custom_dropout_'+str(i)))
            else: # full connection layer
                top_model.add(Dense(n, activation='relu', name='custom_dense_'+str(i)))

        top_model.add(Dense(full_conn[-1], activation='softmax', name='custom_out_'+str(i+1)))
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    return model

def mobilenet_v2_custom_full_conn2(
        in_shape: Tuple[int, int, int],
        weights: Union[None, Text],
        full_conn: List[int],
        alpha: float=1.0
        ):
    '''mobilenet v2 with customized full connection layers
    :param in_shape: input shape (height, width, channel)
    :param full_conn: specify last full connection layers structure except output layer.
    :param alpha: alpha value to adjust the no. of filters
    '''
    model = None
    main_input = Input(shape=in_shape)
    base_model = MobileNetV2(
            include_top=False,
            weights=weights,
            input_tensor=main_input
            )
    
    if full_conn is None:
        raise RuntimeError('invalid arguments, "{}". You must set `full_conn` argument for this network.'.format(full_conn))
    else:
        mbn2_output = base_model.output
        print(mbn2_output)
        x = GlobalAveragePooling2D()(mbn2_output)
        for n in full_conn[:-1]: # except the output layer
            assert n > 0
            if n < 1: # means dropout
                x = Dropout(n)(x)
            else: # full connection layer
                x = Dense(n, activation='relu')(x)

        output = Dense(full_conn[-1], activation='softmax')(x)

    model = Model(inputs=main_input, outputs=output)

    return model

