import pickle
import os

from keras.models import *
from keras.callbacks import *


file_path = os.path.split(os.path.abspath(__file__))[0]


def data_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


data = {
    "x_test": data_load(os.path.abspath(file_path + '/../data/x_test.pkl')),
    "y_test": data_load(os.path.abspath(file_path + '/../data/y_test.pkl'))
}

model = load_model(os.path.abspath(file_path + '/../models/rnn_model_best.pkl'))

########################################################################################################################

BATCH_SIZE = 32

binary_crossentropy, binary_accuracy = model.evaluate(data["x_test"],
                                                      data['y_test'],
                                                      batch_size=BATCH_SIZE)

print('Test binary_crossentropy:', binary_crossentropy)
print('Test binary_accuracy:', binary_accuracy)
