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

(X_train, y_train), _ = mnist.load_data()

IMAGE_MATRIX_ROWS_NUMBER = 28
IMAGE_MATRIX_COLMNS_NUMBER = 28
NUMBER_CLASSES = 10

print('*' * 30, 'Data illustration', '*' * 30)
print('Count objects in train:', len(X_train))
print('Shape of 1 object:', X_train[0].shape)
print('Number of classes:', len(set(y_train)))
# plt.imshow(X_train[0], 'gray')
# print("Digit class:", y_train[0])

print('*' * 30, 'Data preparation', '*' * 30)
x_train = X_train.reshape(X_train.shape[0], IMAGE_MATRIX_ROWS_NUMBER, IMAGE_MATRIX_COLMNS_NUMBER, 1)
input_shape = (IMAGE_MATRIX_ROWS_NUMBER, IMAGE_MATRIX_COLMNS_NUMBER, 1)
print('New train object shape:', x_train[0].shape)

x_train = x_train.astype('float32')
x_train /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')

y_train = to_categorical(y_train, NUMBER_CLASSES)

###################################################################################################################

NUMBER_NEURONS_DENSE_LAYER = 128
OUTPUT_DIMENSION_CONV_1 = 32
OUTPUT_DIMENSION_CONV_2 = 64
SIZE_OF_RECEPTIVE_FIELD = (3, 3)

model = Sequential()
model.add(Conv2D(OUTPUT_DIMENSION_CONV_1,
                 kernel_size=SIZE_OF_RECEPTIVE_FIELD,
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(OUTPUT_DIMENSION_CONV_2,
                 kernel_size=SIZE_OF_RECEPTIVE_FIELD,
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())    # сглаживание входа // smoothing input data
model.add(Dense(NUMBER_NEURONS_DENSE_LAYER, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUMBER_CLASSES, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

####################################################################################################################

BATCH_SIZE = 128
EPOCHS = 10

model_checkpoint = ModelCheckpoint(filepath=os.path.abspath(file_path + '/../models/cnn_model_best.pkl'),
                                   monitor='val_loss',
                                   verbose=0,
                                   save_weights_only=False,
                                   save_best_only=True,
                                   mode='min')

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=0,
                               mode='auto')

model.fit(x_train,
          y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_split=0.1,
          shuffle=True,
          callbacks=[model_checkpoint, early_stopping]
          )
