"""
Deep learning with Keras
model MNIST dataset
"""

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
np.random.seed(1671) # for reproducibility



# network and training
nb_epoch = 250
batch_size = 128
verbose = 1
nb_classes = 10 # number of outputs = number of digits
optimizer = SGD() # SGD optimizer
n_hidden = 128
validation_split = 0.2 # how much TRAIN is reserved for validation
dropout = 0.3
# data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train is 60000 rows of 28x28 values --> reshaped in 60000x784
reshaped = 784

X_train = X_train.reshape(60000, reshaped)
X_test = X_test.reshape(10000, reshaped)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train sample')
print(X_test.shape[0], 'test sample')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


# 10 outputs
# final stage is softmax
model = Sequential()
model.add(Dense(n_hidden, input_shape=(reshaped,)))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(n_hidden))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.summary()

# 激活函数
# simgoid 用于二分类，每个值都介于0和1之间，但每各值之间相互独立
# softmax 用于多分类，每个值都介于0和1之间，所有值相加之和为1

# 损失函数
# binary cross-entropy： 二值交叉熵损失
# -tlog(p) - (1-t)log(1-p)
# t为真实目标，p为预测的概率
# categorical cross-entropy: 多分类交叉熵损失
# 适合多分类，是softmax的默认损失函数

# 度量，只用于测试而不用于训练
# Accuracy
# Precision
# Recall
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                    verbose=verbose, validation_split=validation_split)
score = model.evaluate(X_test, Y_test, verbose=verbose)
print('Test score: ', score[0])
print('Test accuracy: ', score[1])