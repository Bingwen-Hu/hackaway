from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimage.preprocessing import ImageToArrayPreprocessor
from pyimage.preprocessing import SimplePreprocessor
from pyimage.datasets import SimpleDatasetLoader
from pyimage.nn.conv import ShallowNet
from pyimage.nn.conv import LeNet
from pyimage.nn.conv import MiniVGGNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse


class LabelEncoder(object):

    def __init__(self, labels):
        self.data = labels
        self.data_size = len(labels)
        # classes and class_num match
        self.classes_ = sorted(set(labels))
        self.class_num = len(self.classes_)

    def transform(self):
        result = np.zeros([self.data_size, self.class_num])
        for i, d in enumerate(self.data):
            index = self.classes_.index(d)
            result[i][index] = 1
        return result

    def inverse_transform(self, data):
        pass            

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
args = vars(ap.parse_args())

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

trainY = LabelEncoder(trainY).transform()
testY = LabelEncoder(testY).transform()

print('[INFO] compiling model...')
opt = SGD(lr=0.005)
model = model.build(width=32, height=32, depth=3, classes=2)
model.compile(loss='binary_crossentropy', optimizer=opt,
    metrics=['accuracy'])

print('[INFO] training network...')
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    batch_size=32, epochs=100, verbose=1)

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