# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf

from config import FLAGS
from model import build_graph
from preprocess import train_data_iterator, test_data_helper


def train():
    with tf.Session() as sess:
        # initialization
        graph = build_graph(top_k=1)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # multi thread
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # log writer
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/val')

        # restore model
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {}".format(ckpt))

        # training begins
        try:
            while not coord.should_stop():
                for step, (x_batch, y_batch) in enumerate(train_data_iterator()):
                    start_time = time.time()
                    feed_dict = {graph['images']: x_batch,
                                 graph['labels']: y_batch,
                                 graph['keep_prob']: 0.8,
                                 graph['is_training']: True}
                    train_opts = [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']]
                    _, loss_val, train_summary, step = sess.run(train_opts, feed_dict=feed_dict)

                    train_writer.add_summary(train_summary, step)
                    end_time = time.time()
                    print("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
                    # eval stage
                    if step % FLAGS.eval_steps == 0:
                        x_batch_test, y_batch_test = test_data_helper(128)
                        feed_dict = {graph['images']: x_batch_test,
                                     graph['labels']: y_batch_test,
                                     graph['keep_prob']: 1.0,
                                     graph['is_training']: False}
                        test_opts = [graph['accuracy'], graph['merged_summary_op']]
                        accuracy_test, test_summary = sess.run(test_opts, feed_dict=feed_dict)
                        if step > 300:
                            test_writer.add_summary(test_summary, step)
                            print('===============Eval a batch=======================')
                            print('the step {0} test accuracy: {1}'.format(step, accuracy_test))
                            print('===============Eval a batch=======================')
                    # save stage
                    if step % FLAGS.save_steps == 0 and step > FLAGS.min_save_steps:
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name), global_step=graph['global_step'])

        except tf.errors.OutOfRangeError:
            print('==================Train Finished================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name), global_step=graph['global_step'])
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train()