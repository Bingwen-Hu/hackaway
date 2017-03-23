"""
Definition of the neural networks. 

"""
import tensorflow as tf
import common



# Utility functions
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def conv2d(x, W, stride=(1, 1), padding='SAME'):
  return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                      padding=padding)


def max_pool(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
  return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


def convolutional_layers():
    """
    Get the convolutional layers of the model.

    """
    x = tf.placeholder(tf.float32, [None, 60, 160, 3])

    # First layer
    W_conv1 = weight_variable([3, 3, 1, 48])
    b_conv1 = bias_variable([48])
    
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2)) # 30x80

    # Second layer
    W_conv2 = weight_variable([3, 3, 48, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(1, 2), stride=(1, 2)) # 30x40

    # Third layer
    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2)) # 15x20
	"""
    # Fourth layer
    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = bias_variable([256])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool(h_conv4, ksize=(2, 2), stride=(2, 2)) # 4x16
	"""
    return x, h_pool3, [W_conv1, b_conv1,
                        W_conv2, b_conv2,
                        W_conv3, b_conv3]
                       # W_conv4, b_conv4]


def get_training_model():

    x, conv_layer, conv_vars = convolutional_layers()
    
    # Densely connected layer
    W_fc1 = weight_variable([15 * 20 * 128, 512])
    b_fc1 = bias_variable([512])

    conv_layer_flat = tf.reshape(conv_layer, [-1, 15 * 20 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)

    # Output layer
    W_fc2 = weight_variable([512, 5 * len(common.CHARS)])
    b_fc2 = bias_variable([5 * len(common.CHARS)])

    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    return (x, y, conv_vars + [W_fc1, b_fc1, W_fc2, b_fc2])


