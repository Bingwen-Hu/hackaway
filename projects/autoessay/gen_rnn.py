import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


with open("os.txt", encoding='gbk') as f:
    raw_texts = f.read()

chars = sorted(list(set(raw_texts)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))


seq_length = 100
x = []
y = []
for i in range(0, len(raw_texts)-seq_length, seq_length):
    given = raw_texts[i:i+seq_length]
    predict = raw_texts[i+seq_length]
    x.append([char_to_int[char] for char in given])
    y.append([char_to_int[predict]])
print(x[3])


n_patterns = len(x)
n_vocab = len(chars)

x = np.reshape(x, (n_patterns, seq_length, 1))
x = x / float(n_vocab)
y = np_utils.to_categorical(y)

model = Sequential()
model.add(LSTM(128, input_shape=(x.shape[1], x.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(x, y, nb_epoch=10, batch_size=32)

def string_to_index(raw_inputs):
    res = []
    for e in raw_inputs[(len(raw_inputs) - seq_length):]:
        res.append(char_to_int[e])
    return res

def predict_next(input_array):
    x = np.reshape(input_array, (1, seq_length, 1))
    x = x / float(n_vocab)
    y = model.predict(x)
    return y

def y_to_char(y):
    largest_index = y.argmax()
    e = int_to_char[largest_index]
    return e

def generate_article(init, rounds = 50):
    in_string = init.lower()
    for i in range(rounds):
        n = y_to_char(predict_next(string_to_index(in_string)))
        in_string += n
    return in_string

init = "操作系统" * 25
article = generate_article(init)
print(article)