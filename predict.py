#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tensorflow import keras
from typing import Text, Tuple

from get_args import arg_parser_pred
from nets.fcn import fcn

def main(target: Text,
         model_path: Text,
         save_dir: Text = '',
         img_size: Tuple[int,int] = (28,28)):
    '''
    :param network: name of network
    :param target: target fila
    :param model_path: model file path (h5)
    :param save_dir: directory to save prediction result
    :param img_size: input image size (W,H) for the network
    '''

    # arguments validation
    if not os.path.isfile(target):
        raise RuntimeError('target is not file or does not exist, {}'.format(target))
    if not os.path.isfile(model_path):
        raise RuntimeError('model_path is not file or does not exist, {}'.format(model_path))
    if len(save_dir) > 0 and not os.path.isdir(save_dir):
        # if it's empty string, it's assumed that there is no need to save, so skip and proceed the process.
        raise RuntimeError('save_dir is not direcotory or does not exist, {}'.format(save_dir))

    # image preprocessing
    raw_image = Image.open(target)
    # convert to grey scale, resize and convert to numpy array
    image = np.array(raw_image.convert("L").resize(img_size))
    image = image / 255.
    image = np.expand_dims(image, axis=0)
    print('image.shape:',image.shape)

    # load model
    model = keras.models.load_model(model_path)

    # predict
    pred = model.predict(image)
    print('prediction:', pred)

    # plot
    plt.subplot(1, 2, 1)
    plt.imshow(raw_image)
    plt.subplot(1, 2, 2)
    plt.title('Prediction')
    plt.barh(range(0,10),pred[0].tolist())
    plt.yticks(range(0,10),range(0,10))
    plt.xticks([0,0.5,1],[0,0.5,1])
    plt.ylim(9.5,-0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    if len(save_dir) > 0:
        plt.savefig(os.path.join(save_dir, 'prediction.png'))
    plt.show()


if __name__ == '__main__':

    # get args
    parser = arg_parser_pred()
    a = parser.parse_args()

    main(a.target,
         a.model_path,
         save_dir=a.save_dir,
         img_size=(a.img_width, a.img_height))
