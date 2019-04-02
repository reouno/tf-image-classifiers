from tensorflow import keras

def mobilenet_v2(
        in_shape,
        out_units,
        alpha=1.0
        ):
    '''mobilenet v2
    :param in_shape: input shape (height, width, channel)
    :param out_units: number of units in output layer
    :param alpha: alpha value to adjust the no. of filters
    '''

    model = keras.applications.MobileNetV2(
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
