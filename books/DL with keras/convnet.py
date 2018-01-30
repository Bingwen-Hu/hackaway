# convention network in keras

# filters: output numbers
# kernel_size: 3x3 typically

from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt

# define LeNet
class LeNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential()
        # conv => RELU => POOL
        model.add(Conv2D(20, kernel_size=5, padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # conv => RELU => POOL
        model.add(Conv2D(50, kernel_size=5, padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # flatten => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))
        # a softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model


# network and training
nb_epoch = 20
batch_size = 128
verbose = 1
optimizer = Adam()
validation_split = 0.2
IMG_ROWS, IMG_COLS = 28, 28 # input image dimensions
nb_classes = 10 # number of outputs = number of digits
input_shape = (IMG_ROWS, IMG_COLS, 1)

# data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# consider them as float and normalize
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

# we need a 60K x [28 x 28 x 1] shape as input to CONVNET
X_train = X_train[:, :, :, np.newaxis]
X_test = X_test[:, :, :, np.newaxis]
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

# initialize the optimizer and model
model = LeNet.build(input_shape=input_shape, classes=nb_classes)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                    verbose=verbose, validation_split=validation_split)
score = model.evaluate(X_test, y_test, verbose=verbose)
print("test score: ", score[0])
print('test accuracy: ', score[1])

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()