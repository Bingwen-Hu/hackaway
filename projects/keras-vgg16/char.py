from keras import metrics
from keras import losses
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers import Input, Dense, Flatten
from keras.models import Model
import tensorflow as tf
import numpy as np
from keras.layers import advanced_activations
from config import FLAGS
from preprocess import train_data_iterator, test_data_helper




def train():
    # model
    X_shape = (FLAGS.image_height, FLAGS.image_width, 3)
    model = Sequential()
    
    model.add(Conv2D(128, kernel_size=3, padding=1, strides=1, input_shape=X_shape))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # conv => RELU => POOL
    model.add(Conv2D(256, kernel_size=3, padding=1, strides=1))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # flatten => RELU layers
    model.add(Conv2D(512, kernel_size=3, padding=1, strides=1))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(1024, kernel_size=3, padding=1, strides=1))
    model.add(advanced_activations.LeakyReLU(alpha=0.3))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Dense(FLAGS.charset_size)
    model.add(Activation('softmax'))

    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    for epoch in range(FLAGS.num_epochs):
        for i, (X_train, y_train) in enumerate(train_data_iterator()):
            X_train, y_train = np.array(X_train), np.array(y_train)
            loss = model.train_on_batch(X_train, y_train)
            print(f"epoch: {epoch} step: {i} loss: {loss}")

            if i % 100 == 0:
                X_test, y_test = test_data_helper()
                X_test, y_test = np.array(X_test), np.array(y_test)
                score = model.test_on_batch(X_test, y_test)
                print(f"score: {score}")

if __name__ == '__main__':
    train()