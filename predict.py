#!/usr/bin/env python

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tensorflow import keras
from typing import Text, Tuple

from get_args import arg_parser_pred
from nets.fcn import fcn

CLASS_LABEL_FILE = 'class_indices.csv'

def main(target: Text,
         model_path: Text,
         save_dir: Text = '',
         img_size: Tuple[int,int] = (28,28),
         color_mode: Text = 'grayscale',
         num_classes: int = 10
         ):
    '''
    :param network: name of network
    :param target: target file
    :param model_path: model file path (h5)
    :param save_dir: directory to save prediction result
    :param img_size: input image size (W,H) for the network
    :param color_mode: one of 'grayscale', 'rgb', 'rgba'.
    :param num_classes: no. of classes
    '''

    # arguments validation
    if not os.path.isfile(target):
        raise RuntimeError('target is not file or does not exist, {}'.format(target))
    if not os.path.isfile(model_path):
        raise RuntimeError('model_path is not file or does not exist, {}'.format(model_path))
    if len(save_dir) > 0 and not os.path.isdir(save_dir):
        # if it's empty string, it's assumed that there is no need to save, so skip and proceed the process.
        raise RuntimeError('save_dir is not direcotory or does not exist, {}'.format(save_dir))

    # model dir
    model_dir, model_file = os.path.split(model_path)

    # image preprocessing
    raw_image = Image.open(target)
    if color_mode.upper() == 'GRAYSCALE':
        # convert to grey scale, resize and convert to numpy array
        image = np.array(raw_image.convert("L").resize(img_size))
    elif color_mode.upper() == 'RGB':
        image = np.array(raw_image.resize(img_size))
    elif color_mode.upper() == 'RGBA':
        # to be implemented
        raise RuntimeError('the color mode has not been supported yet. "{}"'.format(color_mode))
    else:
        raise RuntimeError('invalid color mode: "{}". "grayscale", "rgb", and "rgba" are supported.'.format(color_mode))
    image = image / 255.
    image = np.expand_dims(image, axis=0)
    print('image.shape:',image.shape)


    # load model
    model = keras.models.load_model(model_path)

    # load class labels
    class_label_file = os.path.join(model_dir, CLASS_LABEL_FILE)
    #class_label_file = 'tmp/mobilenet_v2_0.7_dogs_log_190320/class_indices.csv'
    labels = {}
    if os.path.isfile(class_label_file):
        with open(class_label_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                labels[int(row[0])] = row[1]

    # predict
    pred = model.predict(image)
    print('prediction:', pred)
    print(type(pred), pred.shape)
    pred_index = np.argmax(pred)
    pred_label = labels[pred_index]
    print('The predicted index: "{}", label: "{}"'.format(pred_label, pred_label))

    # plot
    plt.subplot(1, 2, 1)
    plt.imshow(raw_image)
    plt.subplot(1, 2, 2)
    plt.title('Prediction')
    plt.barh(range(0,num_classes),pred[0].tolist())
    plt.yticks(range(0,num_classes),range(0,num_classes))
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
    assert len(a.img_size) == 2 # [W, H]


    main(a.target,
         a.model_path,
         save_dir=a.save_dir,
         img_size=tuple(a.img_size),
         color_mode=a.color_mode,
         num_classes=a.num_classes
         )
