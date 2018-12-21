import os
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model

batch_size = 128
num_epochs = 20
model_dir = "./models"
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784).astype('float32') / 255
X_test = X_test.reshape(10000, 784).astype('float32') / 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)



model = load_model('models/model-05.h5')
# save best model
checkpoint = ModelCheckpoint(filepath=os.path.join(model_dir, 'model-{epoch:02d}.h5'), save_best_only=True)
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=num_epochs,
          validation_split=0.1, callbacks=[checkpoint])

