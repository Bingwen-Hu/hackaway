# basic RNN, have a problem of vanish gradient

from __future__ import print_function
from keras.layers import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
import numpy as np


# text preprocess
with open("./alice_in_wonderland.txt", 'rb') as fin:
    lines = []
    for line in fin:
        line = line.strip().lower()
        line = line.decode('ascii', 'ignore')
        if len(line) == 0:
            continue
        lines.append(line)
text = " ".join(lines)

# lookup table
chars = set(text)
nb_chars = len(chars)
char2index = dict((c, i) for i, c in enumerate(chars))
index2char = dict((i, c) for i, c in enumerate(chars))

# create input and label text
SEQLEN = 10
STEP = 1

input_chars = []
label_chars = []
for i in range(0, len(text) - SEQLEN, STEP):
    input_chars.append(text[i : i+SEQLEN])
    label_chars.append(text[i+SEQLEN])


# transform input and label text to one-hot form
X = np.zeros((len(input_chars), SEQLEN, nb_chars), dtype=np.bool)
y = np.zeros((len(input_chars), nb_chars), dtype=np.bool)
for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        X[i, j, char2index[ch]] = 1
    y[i, char2index[label_chars[i]]] = 1


# define a model
HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 25
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 100

model = Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=False,
                    input_shape=(SEQLEN, nb_chars), unroll=True))
model.add(Dense(nb_chars))
model.add(Activation("softmax"))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


for iteration in range(NUM_ITERATIONS):
    print("=" * 50)
    print("Iteration #: %d" % (iteration))
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)

    test_idx = np.random.randint(len(input_chars))
    test_chars = input_chars[test_idx]
    print("Generationg from seed: %s" % (test_chars))
    print(test_chars, end="")
    for i in range(NUM_PREDS_PER_EPOCH):
        X_test = np.zeros((1, SEQLEN, nb_chars))
        for i, ch in enumerate(test_chars):
            X_test[0, i, char2index[ch]] = 1
        pred = model.predict(X_test, verbose=0)[0]
        y_pred = index2char[np.argmax(pred)]
        print(y_pred, end='')
        # move forward with test_chars + y_pred
        test_chars = test_chars[1:] + y_pred
print()