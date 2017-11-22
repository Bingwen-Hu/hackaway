# -*- coding: utf-8 -*-

import tensorflow as tf

from config import FLAGS
from model import build_graph
from preprocess import train_data_iterator, test_data_helper




with tf.Session() as sess:
