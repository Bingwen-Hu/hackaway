import os
import time
import tensorflow as tf

from config import FLAGS
from Resnet_v2 import build_graph
from preprocess import train_data_iterator, test_data_helper


def train():
    with tf.Session() as sess:
        # initialization
        graph = build_graph(is_training=True)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # multi thread
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

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
                                 graph['labels']: y_batch}
                    _, loss_val = sess.run([graph['optimizer'], graph['loss']], feed_dict=feed_dict)
                    end_time = time.time()
                    print("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))

                    # eval stage
                    if step % FLAGS.eval_steps == 0:
                        x_batch_test, y_batch_test = test_data_helper(128)
                        feed_dict = {graph['images']: x_batch_test,
                                     graph['labels']: y_batch_test}
                        accuracy_test = sess.run(graph['accuracy'], feed_dict=feed_dict)
                        print('===============Eval a batch=======================')
                        print('the step {0} test accuracy: {1}'.format(step, accuracy_test))
                        print('===============Eval a batch=======================')
                    # save stage
                    if step % FLAGS.save_steps == 0 and step > FLAGS.min_save_steps:
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name), global_step=step)
        except tf.errors.OutOfRangeError:
            print('==================Train Finished================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name), global_step=step)
        finally:
            coord.request_stop()
        coord.join(threads)
if __name__ == '__main__':
    train()