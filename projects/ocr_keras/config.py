# -*- coding: utf-8 -*-
""" config for characterOCR
"""
import tensorflow as tf


with open('wordset/words.txt', encoding='utf-8') as fwords:
    words = fwords.read().strip()

WORDSET = words

tf.app.flags.DEFINE_string('wordset', WORDSET, 'Wordset to recognize')
tf.app.flags.DEFINE_integer('wordset_size', len(WORDSET), "Choose the first `charset_size` characters only.")
tf.app.flags.DEFINE_integer('image_size', 64, "Needs to provide same value as in training.")

tf.app.flags.DEFINE_integer('batch_size', 128, 'Validation batch size')
tf.app.flags.DEFINE_integer('num_epochs', 30, 'Validation batch size')
tf.app.flags.DEFINE_integer('min_save_steps', 3000, 'the mininum training steps to save a model')
tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 1000, "the steps to save")

tf.app.flags.DEFINE_string('model_name', 'character-model', 'Name of model file')
tf.app.flags.DEFINE_string('checkpoint_dir', './models/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', './train/original', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', './train/test', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
FLAGS = tf.app.flags.FLAGS