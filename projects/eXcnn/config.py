# -*- coding: utf-8 -*-
""" CNN config for captcha
"""
import string
import tensorflow as tf


CHARSET = string.ascii_letters + string.digits
tf.app.flags.DEFINE_string('charset', CHARSET, "Charset to recognize")
tf.app.flags.DEFINE_integer('charset_size', len(CHARSET), "Charset size obviously")
tf.app.flags.DEFINE_integer('captcha_size', 5, "Number of captcha text")
tf.app.flags.DEFINE_integer('image_height', 40, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_integer('image_width', 100, "Needs to provide same value as in training.")

tf.app.flags.DEFINE_integer('batch_size', 64, 'Validation batch size')
tf.app.flags.DEFINE_integer('num_epochs', 20, 'Validation batch size')
tf.app.flags.DEFINE_integer('min_save_steps', 300, 'the mininum training steps to save a model')
tf.app.flags.DEFINE_integer('eval_steps', 100, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 1000, "the steps to save")

tf.app.flags.DEFINE_string('model_name', 'captcha-model', 'Name of model file')
tf.app.flags.DEFINE_string('checkpoint_dir', './models/', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', 'E:/captcha-data/zips/12W/', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', 'E:/captcha-data/sina/test2/', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from checkpoint')

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
FLAGS = tf.app.flags.FLAGS