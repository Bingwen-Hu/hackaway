# -*- coding: utf-8 -*-
""" config for characterOCR
"""


from __future__ import print_function
import os
import argparse
import numpy as np
from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Model, load_model

from config import FLAGS
from preprocess import test_data_helper, train_data_iterator




def build_model(include_top=True, input_shape=(64, 64, 1), classes=None):
    img_input = Input(shape=input_shape)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dropout(0.05)(x)
        x = Dense(1024, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(img_input, x, name='model')
    return model


model = build_model(classes=FLAGS.wordset_size)

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'],
)



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
            if score[1] > 0.5:
                model.save('model.h5')