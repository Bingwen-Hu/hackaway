# ==================== tensorflow face recognition
import tensorflow as tf
import pandas as pd
import numpy as np
import random

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

def cnn_model():
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    w_conv3 = weight_variable([3, 3, 64, 96])
    b_conv3 = bias_variable([96])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    w_conv4 = weight_variable([3, 3, 96, 128])
    b_conv4 = bias_variable([128])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, w_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)


    # fc mean full connection
    w_fc1 = weight_variable([3 * 3 * 128, 256])
    b_fc1 = bias_variable([256])
    h_pool4_flat = tf.reshape(h_pool4, [-1, 3 * 3 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, w_fc1) + b_fc1)

    # Dropout

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # connect the output layer
    w_fc2 = weight_variable([256, 7])
    b_fc2 = bias_variable([7])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    return y_conv

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

def get_dataset(df):
    df['pixels'] = df['pixels'].apply(lambda img: np.fromstring(img, sep=' ') / 255.0)
    X = np.vstack(df['pixels'])
    X = X.reshape((-1, 48, 48, 1))
    y = one_hot_encode(df['emotion'])
    return X, y

dataset = pd.read_csv("/home/mory/kaggle/fer/fer2013/fer2013.csv", nrows=10000)

index = list(range(dataset.shape[0]))
random.shuffle(index)

X, y = get_dataset(dataset)
X_train = X[index][:9000]
y_train = y[index][:9000]
X_test = X[index][9000:]
y_test = y[index][9000:]

X = None
y = None

x = tf.placeholder(tf.float32, [None, 48, 48, 1])
y_ = tf.placeholder(tf.float32, [None, 7])
keep_prob = tf.placeholder(tf.float32)

y_conv = cnn_model()
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

BATCH_SIZE = 100
TRAIN_SIZE = X_train.shape[0]
train_index = list(range(TRAIN_SIZE))
saver = tf.train.Saver()
best_accuracy = 0
stop = 0
print('begin training..., train dataset size:{0}'.format(TRAIN_SIZE))
outputfile = open('log.txt', 'w')
print("train size: %d  test size: %d" % (TRAIN_SIZE, X_test.shape[0]), 
      file=outputfile)
for i in range(100):
    random.shuffle(train_index)  #每个epoch都shuffle一下效果更好
    X_train, y_train = X_train[train_index], y_train[train_index]

    for j in range(0,TRAIN_SIZE,BATCH_SIZE):
        print('epoch {0}, train {1} samples done...'.format(i,j))
        train_step.run(feed_dict={x:X_train[j:j+BATCH_SIZE], 
                                  y_:y_train[j:j+BATCH_SIZE], keep_prob:0.5})

    current_accuracy = accuracy.eval(feed_dict={x:X_test, y_:y_test, keep_prob: 1.0})
    if best_accuracy < current_accuracy:
        best_accuracy = current_accuracy
 #       current_epoch = i
 #       saver.save(sess, 'face_model')
        stop = 0
    else:
        stop += 1
        if stop>10:
            break
    print("the accracy is: %.3f" % current_accuracy)
    print("epoch %d, accuracy: %.3f" % (i, current_accuracy), file=outputfile)
print("best accuracy: %.3f" % best_accuracy, file=outputfile)
