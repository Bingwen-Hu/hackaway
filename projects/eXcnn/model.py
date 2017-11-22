""" CNN model architecture for captcha
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

from config import FLAGS


def build_graph():
    # define placeholder
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, name='input_image_batch',
        shape=[None, FLAGS.image_height * FLAGS.image_width])
    x_images = tf.reshape(images, name="image_batch",
        shape=[-1, FLAGS.image_height, FLAGS.image_width, 1])
    labels = tf.placeholder(dtype=tf.float32, name='label_batch',
        shape=[None, FLAGS.captcha_size * FLAGS.charset_size])

    # define convention network
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm):
        conv3_1    = slim.conv2d(x_images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
        max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], 2, padding='SAME', scope='pool1')
        conv3_2    = slim.conv2d(max_pool_1, 128, [3, 3], 1, padding='SAME', scope='conv3_2')
        max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], 2, padding='SAME', scope='pool2')
        conv3_3    = slim.conv2d(max_pool_2, 256, [3, 3], 1, padding='SAME', scope='conv3_3')
        max_pool_3 = slim.max_pool2d(conv3_3, [2, 2], 2, padding='SAME', scope='pool3')
        conv3_4    = slim.conv2d(max_pool_3, 512, [3, 3], 1, padding='SAME', scope='conv3_4')
        conv3_5    = slim.conv2d(conv3_4, 512, [3, 3], 1, padding='SAME', scope='conv3_5')
        max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], 2, padding='SAME', scope='pool4')
        flatten    = slim.flatten(max_pool_4)
        fc1        = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024,
                        activation_fn=tf.nn.relu, scope='fc1')
        logits     = slim.fully_connected(slim.dropout(fc1, keep_prob),
                        FLAGS.charset_size * FLAGS.captcha_size , activation_fn=None, scope='fc2')

    # define loss and optimizer
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # define accuracy
    logits_indices = tf.argmax(
        tf.reshape(logits, [-1, FLAGS.captcha_size, FLAGS.charset_size]), 2)
    labels_indices = tf.argmax(
        tf.reshape(labels, [-1, FLAGS.captcha_size, FLAGS.charset_size]), 2)
    accuracy = tf.reduce_mean(tf.cast(
        tf.equal(logits_indices, labels_indices), tf.float32))
    return images, labels, keep_prob, loss, optimizer, accuracy