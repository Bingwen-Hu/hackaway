# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf

import model

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='E:/captcha-data/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')
parser.add_argument('--model_dir', type=str, default="./models/",
                    help="The directory where the model will be stored.")
parser.add_argument('--resent_size', type=int, default=32,
                    help='The size of the ResNet model to use.')
parser.add_argument('--train_epochs', type=int, default=250,
                    help='The number of epochs to train.')
parser.add_argument('--epochs_per_eval', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')
parser.add_argument('--data_format', type=str, default=None,
                    choices=['channels_first', 'channels_last'],
                    help='A flag to override the data format used in the model. channels_first'
                        'provides a performance boost on GPU but is not always compatible with '
                        'CPU. If left unspecified, the data format will be chosen automatically '
                        'based on whether Tensorflow was bulit for CPU or GPU.')

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

# we use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train':50000,
    'validation':10000,
}

# newer API
def record_dataset(filenames):
    """Returns an input pipeline Dataset from `filenames`."""
    record_bytes = _HEIGHT * _WIDTH * _DEPTH + 1    # 1 is label bytes
    return tf.data.FixedLengthRecordDataset(filenames, record_bytes)


def get_filenames(is_training, data_dir):
    """Return a list of filenames."""
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

    assert os.path.exists(data_dir), (
            "Run cifar10_download_and_extract.py first to download and extract the "
            "CIFAR-10 data")

    if is_training:
        return [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                for i in range(1, _NUM_DATA_FILES + 1)]
    else:
        return [os.path.join(data_dir, 'test_batch.bin')]


def parse_record(raw_record):
    """Parse CIFAR-10 image and label from a raw record."""
    # Every record consists of a label followed by the image, with a fixed number
    # of bytes for each.
    label_bytes = 1
    image_bytes = _HEIGHT * _WIDTH * _DEPTH
    record_bytes = label_bytes + image_bytes

    # Convert bytes to a vector of uint8 that is record_bytes long.
    record_vector = tf.decode_raw(raw_record, tf.uint8)

    # The first byte represents the label, which we convert from uint8 to int32
    # and then to one-hot.
    label = tf.cast(record_vector[0], tf.int32)
    label = tf.one_hot(label, _NUM_CLASSES)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width]
    depth_major = tf.reshape(record_vector[label_bytes:record_bytes],
                             [_DEPTH, _HEIGHT, _WIDTH])
    # why bother?
    # Convert from [depth, height, width] to [height, width, depth], and cast
    # as float32
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    return image, label


def preprocess_image(image, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
                image, _HEIGHT + 8, _WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
    """Input_fn using thet tf.data input pipeline for CIFAR-10 dataset.
    Returns:
        A tuple of images and labels
    """
    dataset = record_dataset(get_filenames(is_training, data_dir))

    if is_training:
        dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

    dataset = dataset.map(parse_record)
    dataset = dataset.map(
            lambda image, label: (preprocess_image(image, is_training), label))
    dataset = dataset.prefetch(2 * batch_size)

    # we call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)

    # batch results by up to batch_size, and then fetch the tuple from the
    # iterator.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels


def cifar10_model_fn(features, labels, mode, params):
    tf.summary.image('images', features, max_outputs=6)

    network = model.cifar10_resnet_v2_generator(
            params['resnet_size'], _NUM_CLASSES, params['data_format'])

    inputs = tf.reshape(features, [-1, _HEIGHT, _WIDTH, _DEPTH])
    logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    pass