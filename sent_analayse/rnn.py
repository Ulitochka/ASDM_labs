import pickle
import os

from keras.optimizers import *
from keras.models import *
from keras.layers import *
from keras.callbacks import *
import tensorflow as tf
from keras import backend as K
import numpy


file_path = os.path.split(os.path.abspath(__file__))[0]


def data_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


data = {
    "x_train": data_load(os.path.abspath(file_path + '/../data/x_train.pkl')),
    "y_train": data_load(os.path.abspath(file_path + '/../data/y_train.pkl'))
}

########################################################################################################################

LSTM_UNITS = 32
OUTPUT_UNITS = 1
RANDOM_EMBEDDING_SIZE = 64
MAX_SENT_LENGTH = 300
MAX_WORDS = 1000

input_sequence = Input((MAX_SENT_LENGTH,))
model = Sequential()
model.add(Embedding(
    input_dim=MAX_WORDS,
    output_dim=64,
    input_length=MAX_SENT_LENGTH,
    mask_zero=True,
    trainable=True))
model.add(LSTM(LSTM_UNITS, return_sequences=True))
model.add(LSTM(LSTM_UNITS))
model.add(Dropout(0.9))
model.add(Dense(OUTPUT_UNITS, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
print(model.summary())

########################################################################################################################

BATCH_SIZE = 32
EPOCH = 10


model_checkpoint = ModelCheckpoint(filepath=os.path.abspath(file_path + '/../models/rnn_model_best.pkl'),
                                   monitor='val_loss',
                                   verbose=0,
                                   save_weights_only=False,
                                   save_best_only=True,
                                   mode='min')

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=0,
                               mode='auto')

history = model.fit(data["x_train"],
                    data["y_train"],
                    batch_size=BATCH_SIZE,
                    epochs=EPOCH,
                    validation_split=0.1,
                    verbose=1,
                    shuffle=True,
                    callbacks=[model_checkpoint, early_stopping])