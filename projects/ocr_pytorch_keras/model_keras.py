from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
import tensorflow as tf
import numpy as np
from config import args
from preprocess import train_data_iterator, test_data_helper

def build_net(args, channels=1):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(args.image_size, args.image_size, channels)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(args.wordset_size))
    model.summary()
    return model


def sigmoid_loss(logits, labels):
    labels_ = tf.squeeze(labels, axis=0)
    labels_ = tf.cast(labels_, tf.int64)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_))
    return loss


def captcha_accuracy(logits, labels):
    labels_ = tf.squeeze(labels, axis=0)
    labels_ = tf.cast(labels_, tf.int64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels_), tf.float32))
    return accuracy


def train():
    model = build_net(args, 1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    for epoch in range(args.epochs):
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
