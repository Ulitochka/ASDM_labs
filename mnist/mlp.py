import os

import numpy
numpy.random.seed(1024)

from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras.callbacks import *
import tensorflow as tf
from keras import backend as K


file_path = os.path.split(os.path.abspath(__file__))[0]


(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('*' * 30, 'Data illustration', '*' * 30)
print('Count objects in train:', len(X_train))
print('Count objects in test:', len(X_test))
print('Shape of 1 object:', X_test[0].shape)
print('Number of classes:', len(set(y_train)))
# plt.imshow(X_train[0], 'gray')
# print("Digit class:", y_train[0])

print('*' * 30, 'Data preparation', '*' * 30)
x_train = X_train.reshape(60000, 784).astype('float32')
x_test = X_test.reshape(10000, 784).astype('float32')
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print('Shape of 1 object:', x_train[0].shape)

###################################################################################################################

FEATURE_VECTOR = 784
NUMBER_NEURONS_HIDDEN_LAYER = 512
NUMBER_NEURONS_OUTPUT_LAYER = 10


model = Sequential()
model.add(Dense(NUMBER_NEURONS_HIDDEN_LAYER, activation='relu', input_shape=(FEATURE_VECTOR,)))
model.add(Dropout(0.2))
model.add(Dense(NUMBER_NEURONS_HIDDEN_LAYER, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(NUMBER_NEURONS_HIDDEN_LAYER, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(NUMBER_NEURONS_HIDDEN_LAYER, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(NUMBER_NEURONS_OUTPUT_LAYER, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

###################################################################################################################

BATCH_SIZE = 128
EPOCHS = 35

model_checkpoint = ModelCheckpoint(filepath=os.path.abspath(file_path + '/../models/mlp_model_best.pkl'),
                                   monitor='val_loss',
                                   verbose=0,
                                   save_weights_only=False,
                                   save_best_only=True,
                                   mode='min')

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=0,
                               mode='auto')

history = model.fit(x_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=[model_checkpoint, early_stopping])



