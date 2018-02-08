from keras.applications import ResNet50
from keras.layers import Input, Dense, Flatten
from keras.models import Model
import tensorflow as tf
import numpy as np

from config import FLAGS
from preprocess import train_data_iterator, test_data_helper




def sigmoid_loss(y_true, y_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)


def captcha_accuracy(y_true, y_pred):
    predict = tf.reshape(y_pred, [-1, FLAGS.captcha_size, FLAGS.charset_size])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(y_true, [-1, FLAGS.captcha_size, FLAGS.charset_size]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))



X_shape = (200, 200, 3)
X_tensor = Input(shape=X_shape)
resnet = ResNet50(include_top=False, weights=None, input_tensor=X_tensor,
                  input_shape=X_shape, pooling=None)
flatten = Flatten()(resnet.output)
output = Dense(FLAGS.charset_size * FLAGS.captcha_size)(flatten)
model = Model(inputs=[X_tensor], outputs=[output])
model.compile(optimizer='adam', loss=sigmoid_loss, metrics=[captcha_accuracy])


for _ in range(FLAGS.num_epochs):
    for i, (X_train, y_train) in enumerate(train_data_iterator()):
        X_train, y_train = np.array(X_train), np.array(y_train)
        loss = model.train_on_batch(X_train, y_train)
        print(f"loss: {loss}")
        
        if i % 1000 == 0:
            X_test, y_test = test_data_helper(128)
            X_test, y_test = np.array(X_test), np.array(y_test)
            score = model.test_on_batch(X_test, y_test)
            print(f"score: {score}")