from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K


# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output                       # (None, 8, 8, 2048
x = GlobalAveragePooling2D()(x)             # (None, 2048)
x = Dense(1024, activation='relu')(x)       # another syntax?
predictions = Dense(200, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# freeze all convolutional inceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# model.fit_generator(...)


# freeze some layer and train other
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True


# recompile and train
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
# model.fit_generator ....
