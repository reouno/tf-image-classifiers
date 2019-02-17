import os
import tensorflow as tf
from tensorflow import keras

class CollectBatchStats(keras.callbacks.Callback):
    '''
    callback to collect training logs
    '''

    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])

    def on_epoch_end(self, batch, logs=None):
        # do something every epoch
        pass

def modelCheckpoint(save_dir):
    '''
    callback to save weights for every 5 epochs
    '''

    ckpt_path = os.path.join(save_dir, 'cp-{epoch:04d}.ckpt')
    cp_callback = keras.callbacks.ModelCheckpoint(ckpt_path,
                                                  save_weights_only=True,
                                                  verbose=1,
                                                  period=5)
    return cp_callback

def tensorboard(log_dir):
    '''
    callback to save stats for tensorboard
    '''
    callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    return callback
