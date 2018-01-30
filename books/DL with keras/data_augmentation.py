from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import load_model, model_from_json
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np


BATCH_SIZE = 128
NB_EPOCHS = 20
NB_CLASSES = 10
OPTIMIZER = Adam()
# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape: ', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert to categorical
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# float and normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


# augmenting
print("Augmenting training set images...")
datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                             zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# fit the datagen
datagen.fit(X_train)

# load the model
with open("cifar10_architecture.json") as f:
    model = model_from_json(f.read())
# model = model_from_json()
model.load_weights('cifar10_weights.h5')
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
# train
history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                              epochs=NB_EPOCHS, verbose=1)
score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=1)
print("test score: ", score[0])
print('test accuracy: ', score[1])