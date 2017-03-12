import tensorflow as tf
import faceRecognize as fr

sess = tf.InteractiveSession()

# step 1: define the struction of network
in_units = 2304
h1_units = 300
w1 = fr.weight_variable([in_units, h1_units])
b1 = fr.bias_variable([h1_units])
w2 = fr.weight_variable([h1_units, 7])
b2 = fr.bias_variable([7])

# using Dropout trick
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, w2) + b2)


# step 2: define the loss function
y_ = tf.placeholder(tf.float32, [None, 7])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), 
                                              reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

# ==================== prepare the data =========================
import pickle
with open('5000data.bat', 'rb') as f:
    train = pickle.load(f)
X_train = fr.str2int(train.X)
y_train = fr.one_hot_encode(train.y)
import pandas as pd
test = pd.read_csv("/home/mory/kaggle/fer/fer2013/fer2013.csv", 
                   nrows=200, skiprows=30000, usecols=[0, 1], names=['y', 'X'])
X_test = fr.str2int(fr.str2list(test.X))
y_test = fr.one_hot_encode(test.y)
# ===============================================================
# step 3: training
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = fr.get_random_block(X_train, y_train, 100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})

# step 4: evaluate
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: X_test, y_: y_test, 
                     keep_prob: 1.0}))
