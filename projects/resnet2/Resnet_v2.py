import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.
    Args:
        scope: block name
        unif_fn:
        args: a list like [[(256, 64, 1], (256, 64, 1), (256, 64, 2)]
            where (256, 64, 3) means the last layer output is 256, pre-layer output is 64,
            and stride in middle layer is 3
    """


def subsample(inputs, factor, scope=None):
    """Subsample using max pool
    :param inputs: inputs
    :param factor: int, using as stride in max pool
    :param scope: scope name
    :return:
    """
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)

    else:
        pad_total = kernel_size - 1
        pad_begin = pad_total // 2
        pad_end   = pad_total - pad_begin
        inputs    = tf.pad(inputs, [[0, 0], [pad_begin, pad_end],     # for input is [None, height, width, channels]
                                    [pad_begin, pad_end], [0, 0]])    # only height and width need padding
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding="VALID", scope=scope)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i+1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net,
                                        depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    return net


def resnet_arg_scope(is_training=True, weight_decay=0.0001, batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding="SAME") as arg_sc:
                return arg_sc


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride,
               outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')

        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None,
                               activation_fn=None, scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)



def resnet_v2(inputs, blocks, num_classes, keep_prob, include_root_block=True, reuse=None, scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense], outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = stack_blocks_dense(net, blocks)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            # glabal pool
            net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            # below codes add by mory
            net = slim.flatten(net)
            net = slim.fully_connected(slim.dropout(net, keep_prob), 1024,
                                       activation_fn=tf.nn.relu, scope='fc1')
            logits = slim.fully_connected(slim.dropout(net, keep_prob),
                                          num_classes, activation_fn=None, scope='logits')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            return logits, end_points


def resnet_v2_50(inputs, num_classes, keep_prob, reuse=None, scope='resnet_v2_50'):
    blocks = [
        Block('block1', bottleneck, [(256,  64,  1)] * 2 + [(256,  64,  2)]),
        Block('block2', bottleneck, [(512,  128, 1)] * 3 + [(512,  128, 2)]),
        Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
        Block('block4', bottleneck, [(2048, 512, 1)] * 3)
    ]
    return resnet_v2(inputs, blocks, num_classes, keep_prob, include_root_block=True,
                     reuse=reuse, scope=scope)

# dirty function
def build_graph(is_training):
    from config import FLAGS
    inputs = tf.placeholder(tf.float32, [None, FLAGS.image_height * FLAGS.image_width], 'images')
    keep_prob = tf.placeholder(tf.float32, [], 'keep_prob')
    inputs_shape = tf.reshape(inputs, [-1, FLAGS.image_height, FLAGS.image_width, 1], 'inputs_shape')
    labels = tf.placeholder(tf.float32, [None, FLAGS.charset_size * FLAGS.captcha_size], 'labels')
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        logits, end_points = resnet_v2_50(inputs_shape, FLAGS.charset_size * FLAGS.captcha_size, keep_prob)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    max_idx_p = tf.argmax(tf.reshape(logits, [-1, FLAGS.charset_size, FLAGS.captcha_size]), 2)
    max_idx_l = tf.argmax(tf.reshape(labels, [-1, FLAGS.charset_size, FLAGS.captcha_size]), 2)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(max_idx_l, max_idx_p), tf.float32))
    return {'images': inputs,
            'labels': labels,
            'accuracy': accuracy,
            'optimizer': optimizer,
            'loss': loss,
            'logits': logits,
            'keep_prob': keep_prob}


if __name__ == '__main__':
    import numpy as np
    from preprocess import train_data_iterator
    batch_size = 32
    height, width = 40, 100
    inputs = tf.placeholder(tf.float32, [None, height * width], 'inputs')
    inputs_shape = tf.reshape(inputs, [-1, height, width, 1], 'inputs_shape')
    labels = tf.placeholder(tf.float32, [None, 310], 'labels')
    keep_prob = tf.placeholder(tf.float32, [], 'keep_prob')
    with slim.arg_scope(resnet_arg_scope(is_training=True)):
       logits, end_points = resnet_v2_50(inputs_shape, 310, keep_prob)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    for X, y in train_data_iterator():
        break
    print(y)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    loss_val = sess.run(loss, feed_dict={inputs: X, labels:y, keep_prob:0.5})
    print("loss: {}".format(loss_val))