from tensorflow import keras
from typing import Text, Tuple

class Classifier:
    def __init__(self, name: Text, in_shape: Tuple, num_class: int):
        '''
        :param in_shape: input shape
        :param num_class: no. of classes, or output units
        '''
