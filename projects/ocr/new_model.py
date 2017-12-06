""" CNN model architecture for captcha
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops

from config import FLAGS


def build_graph(top_k):
    # define placeholder
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, name='input_image_batch',
        shape=[None, FLAGS.image_size * FLAGS.image_size])
    x_images = tf.reshape(images, name="image_batch",
        shape=[-1, FLAGS.image_size, FLAGS.image_size, 1])
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')

    # define convention network
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training':is_training}):
        conv3_1    = slim.conv2d(x_images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
        max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], 2, padding='SAME', scope='pool1')
        conv3_2    = slim.conv2d(max_pool_1, 128, [3, 3], 1, padding='SAME', scope='conv3_2')
        max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], 2, padding='SAME', scope='pool2')
        # mory add conv3m2 and max_poolm2
        conv3m2    = slim.conv2d(max_pool_2, 128, [3, 3], 1, padding='SAME', scope='conv3m2')
        max_poolm2 = slim.max_pool2d(conv3m2, [2, 2], 2, padding='SAME', scope='pool2m')
        conv3_3    = slim.conv2d(max_poolm2, 256, [3, 3], 1, padding='SAME', scope='conv3_3')
        max_pool_3 = slim.max_pool2d(conv3_3, [2, 2], 2, padding='SAME', scope='pool3')
        conv3_4    = slim.conv2d(max_pool_3, 512, [3, 3], 1, padding='SAME', scope='conv3_4')
        conv3_5    = slim.conv2d(conv3_4, 512, [3, 3], 1, padding='SAME', scope='conv3_5')
        max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], 2, padding='SAME', scope='pool4')
        flatten    = slim.flatten(max_pool_4)
        fc1        = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024,
                        activation_fn=tf.nn.relu, scope='fc1')
        logits     = slim.fully_connected(slim.dropout(fc1, keep_prob),
                        FLAGS.wordset_size, activation_fn=None, scope='fc2')

    # define loss and optimizer
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        loss = control_flow_ops.with_dependencies([updates], loss)
    # ugly code
    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = slim.learning.create_train_op(loss, optimizer, global_step=global_step)
    probabilities = tf.nn.softmax(logits)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k)
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_k), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_k,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'is_training': is_training,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}