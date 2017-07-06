import os

import numpy
numpy.random.seed(1024)

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import *
from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras.callbacks import *
import tensorflow as tf
from keras import backend as K


file_path = os.path.split(os.path.abspath(__file__))[0]

IMAGE_MATRIX_ROWS_NUMBER = 28
IMAGE_MATRIX_COLMNS_NUMBER = 28
NUMBER_CLASSES = 10

_, (X_test, y_test) = mnist.load_data()

print('*' * 30, 'Data preparation', '*' * 30)
x_test = X_test.reshape(X_test.shape[0], IMAGE_MATRIX_ROWS_NUMBER, IMAGE_MATRIX_COLMNS_NUMBER, 1)
input_shape = (IMAGE_MATRIX_ROWS_NUMBER, IMAGE_MATRIX_COLMNS_NUMBER, 1)
print('New train object shape:', x_test[0].shape)

x_train = x_test.astype('float32')
x_train /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

y_test = to_categorical(y_test, NUMBER_CLASSES)

model = load_model(os.path.abspath(file_path + '/../models/cnn_model_best.pkl'))

########################################################################################################################

BATCH_SIZE = 128

categorical_crossentropy, accuracy = model.evaluate(x_test,
                                                    y_test,
                                                    verbose=0,
                                                    batch_size=BATCH_SIZE)

print('Test categorical_crossentropy:', categorical_crossentropy)
print('Test accuracy:', accuracy)

predictions = model.predict_classes(x_test, verbose=0)
error = 0
for i in range(0, len(y_test)):
    if list(y_test[i]).index(1) != predictions[i]:
        error += 1
print('Errors:', error, 'All test instance:', len(y_test))
