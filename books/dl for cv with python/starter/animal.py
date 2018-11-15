from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimage.preprocessing import ImageToArrayPreprocessor
from pyimage.preprocessing import SimplePreprocessor
from pyimage.datasets import SimpleDatasetLoader
from pyimage.nn.conv import ShallowNet
from pyimage.nn.conv import LeNet
from pyimage.nn.conv import MiniVGGNet
from pyimage.callbacks import TrainingMonitor
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


NETWORK_BANK = {
    'shallownet': ShallowNet,
    'minivggnet': MiniVGGNet,
    'lenet': LeNet,
}

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
    help='path to input dataset')
ap.add_argument('-n', '--network', required=True,
    help='model to test %s' % ', '.join(NETWORK_BANK.keys()))
ap.add_argument('-o', '--output', required=True,
    help='path to the output directory')
ap.add_argument('-w', '--weights', required=True,
    help='path to best weight file')
args = vars(ap.parse_args())

# easily tracking
print('[INTO] process ID: {}'.format(os.getpid()))

print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))
model = NETWORK_BANK[args['network']]

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype('float') / 255.0

trainX, testX, trainY, testY = train_test_split(data, labels,
    test_size=0.25, random_state=42)

lb = LabelBinarizer()
trainY = np_utils.to_categorical(lb.fit_transform(trainY), 2)
testY = np_utils.to_categorical(lb.fit_transform(testY), 2)

print('[INFO] compiling model...')
opt = SGD(lr=0.005)
model = model.build(width=32, height=32, depth=3, classes=2)
model.compile(loss='binary_crossentropy', optimizer=opt,
    metrics=['accuracy'])

# plot-log
logdir = args['output']
if not os.path.exists(logdir):
    os.makedirs(logdir)
figPath = os.path.sep.join([args['output'], '{}.png'.format(os.getpid())])
jsonPath = os.path.sep.join([args['output'], '{}.json'.format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

# checkpoint: for illustration
# checkpoint = ModelCheckpoint(args['weights'], monitor='val_loss', mode='min', 
#     save_best_only=True, verbose=1)
# callbacks.append(checkpoint)

print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    batch_size=32, epochs=100, verbose=2, callbacks=callbacks)

print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=['cat', 'dog']))


plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()