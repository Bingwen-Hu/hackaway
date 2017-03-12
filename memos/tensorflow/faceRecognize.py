# ==================== tensorflow face recognition
import tensorflow as tf
import pandas as pd
import numpy as np

def str2list(X, number=None):
    if not number:
        number = len(X)
    X_ = [x.split(' ') for x in X[:number]]
    return X_

def str2int(X):
    X_ = [list(map(int, x)) for x in X]
    return X_

# function for convenience
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

# downsample 2x2 to 1x1
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding='SAME')

# random batch for training
def get_random_block(X, y, batch_size):
    start_index = np.random.randint(0, len(X) - batch_size)
    return X[start_index: (start_index + batch_size)], y[start_index: (start_index + batch_size)]

def one_hot_encode(X):
    column = len(set(X))
    row = len(X)
    new_X = np.zeros((row, column))
    for x_, x in zip(new_X, X):
        x_[x] = 1
    return new_X

# prepare the train set and test set
def get_trainset():
    import pickle
    all_df = pd.DataFrame(columns=['X', 'y'])
    filenames = ['%sdata.bat' % s for s in ['5000', '10000', '15000', '20000']]
    for filename in filenames:
        with open(filename, "rb") as f:
            df = pickle.load(f)
        all_df = all_df.append(df)
    return all_df
"""
dataset = pd.read_csv("/home/mory/kaggle/fer/fer2013/fer2013.csv", names=['pixels', 'emotion'],
                      nrows=100, skiprows=30000)
X_test = dataset.pixels
y_test = dataset.emotion
# transform
X_test = str2list(X_test)
X_test = str2int(X_test)
y_test = one_hot_encode(y_test)
"""

"""
train = dataset[dataset.Usage=='Training']
test = dataset[dataset.Usage=='PrivateTest'


print(train.shape, test.shape)

X_train = train.pixels
y_train = train.emotion
X_test = test.pixels
y_test = test.emotion


# prepare the training data
X_tr = str2list(X_train)
X_tr = str2int(X_tr)             # list 
y_tr = one_hot_encode(y_train)      # array

X_te = str2list(X_test, 100)
X_te = str2int(X_te)
y_te = one_hot_encode(y_test[:100])
# now m=2000 p=2304 channel=1, output class=7
"""

x = tf.placeholder(tf.float32, [None, 2304])
y_ = tf.placeholder(tf.float32, [None, 7])
x_image = tf.reshape(x, [-1, 48, 48, 1])

# define the first net
w_conv1 = weight_variable([5, 5, 1, 6])
b_conv1 = bias_variable([6])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

w_conv2 = weight_variable([12, 12, 6, 14])
b_conv2 = bias_variable([14])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# fc mean full connection
w_fc1 = weight_variable([12 * 12 * 14, 128])
b_fc1 = bias_variable([128])
h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 14])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# connect the output layer
w_fc2 = weight_variable([128, 7])
b_fc2 = bias_variable([7])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# training
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), 
                                              reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# evaluate
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# commit training
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()



def commit_train(times=100):
    for i in range(times):
        b_x, b_y = get_random_block(X_train, y_train, 50)
        train_step.run(feed_dict={x: b_x, y_: b_y, keep_prob: 0.5})
    print("test accuracy %g" % accuracy.eval(feed_dict={x: X_test, y_: y_test,
                                                    keep_prob: 1.0}))
