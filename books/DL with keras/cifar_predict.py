import numpy as np
import imageio
import skimage
from keras.models import model_from_json
from keras.optimizers import SGD


# load model
model_architecture = 'cifar10_architecture.json'
model_weights = 'cifar10_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)


# load images
img_names = ['/home/mory/Downloads/cat.jpeg', '/home/mory/Downloads/dog.jpg']
imgs = [np.transpose(skimage.transform.resize(imageio.imread(img_name), (32, 32)),
                     (1, 0, 2)).astype('float32') for img_name in img_names]
imgs = np.array(imgs) / 255

# train
optimizer = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# predict
predictions = model.predict_classes(imgs)
print(predictions)