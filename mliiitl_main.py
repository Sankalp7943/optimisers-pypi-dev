import numpy as np
import matplotlib as plt
import keras
import tensorflow as tf
import sys

class mliiitl:
    '''
    Creates mliiitl object from all the user data
    '''
    def __init__(self, x_train, y_train, x_test, y_test, 
                 model, loss, epoch, batch_size):
        try:
            self._x_train = x_train
            self._y_train = y_train
            self._x_test = x_test
            self._y_test = y_test
            self._model = model
            self._loss = loss
            self._epoch = epoch
            self._batch_size = batch_size
        except Exception:
            try:
                print('Invalid arguments given in mliiitl.__init__()')
            except Exception:
                pass
            try:
                print('Invalid arguments given in mliiitl.__init__()', file = sys.stdout)
            except Exception:
                pass 
        return self
        
        

