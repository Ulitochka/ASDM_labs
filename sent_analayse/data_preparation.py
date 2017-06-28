import os
import re
import pandas as pd
import pickle
from random import shuffle
from pprint import pprint
import codecs
import csv

import numpy

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


MAX_WORDS = 1000
MAX_SENT_LENGTH = 300
tokenizer = Tokenizer(num_words=MAX_WORDS)


file_path = os.path.split(os.path.abspath(__file__))[0]
data_path = os.path.abspath(file_path + '/../data/data.csv')


def load_simple(path):
    """
    Data items: 2500
    :param path: 
    :return: 
    """
    x = []
    y = []
    f = open(path, encoding='utf-8')
    data = f.readlines()
    for index, line in enumerate(data):
        if index > 0:
            tokens = line.strip().split()
            x.append(tokens[1:])
            y.append(y)
    return x, y


def clean(s):
    return re.sub(r'[^\x00-\x7f]', r'', s)


def clean_html(html):
    p = re.compile(r'<.*?>')
    return p.sub('', html)


def load_with_clean(path):
    data = pd.read_csv(path, header=0, delimiter="\t", quoting=3)
    txt = ''
    docs = []
    sentences = []
    sentiments = []
    for cont, sentiment in zip(data.text, data.sentiment):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', clean(clean_html(cont)))
        sentences = [sent.lower() for sent in sentences]
        docs.append(sentences)
        sentiments.append(sentiment)
    # 1 docs = 1 concatenate string
    docs = [' '.join(d) for d in docs]
    return docs, sentiments


def token_to_index(x):
    """
    Because we delete words with freq less 1000, we have different lengths:
    :param x: 
    :return: 
    """

    # sent = [sent for docs in x for sent in docs]
    # tokenizer.fit_on_texts(sent)
    tokenizer.fit_on_texts(x)
    x = tokenizer.texts_to_sequences(x)
    return x


def padding_sent(data):
   return pad_sequences(data, maxlen=MAX_SENT_LENGTH)


def test_train_split(x, y):
    return numpy.array(x[:2000]), numpy.array(x[-500:]), numpy.array(y[:2000]), numpy.array(y[-500:])

def save_binary(data, file_name):
    with open(os.path.abspath(file_path + '/../data/%s.pkl') % file_name, 'wb') as file:
        pickle.dump(data, file)


x, y = load_with_clean(data_path)
x_indexing = token_to_index(x)
x_padding = padding_sent(x_indexing)
x_train, x_test, y_train, y_test = test_train_split(x_padding, y)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

save_binary(x_train, 'x_train')
save_binary(x_test, 'x_test')
save_binary(y_train, 'y_train')
save_binary(y_test, 'y_test')
