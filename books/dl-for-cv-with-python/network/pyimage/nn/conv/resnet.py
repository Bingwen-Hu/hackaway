from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K



class ResNet:
    @staticmethod
    def residual_module(data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
        """
        Args:
            data: input
            K: number of output, aka filters
            stride: convolution stride
            chanDim: channel dimension
            red: sign whether this module response to reduce spatial dimension
            bnEps: bn layer eps
            bnMon: bn layer momentum
        """
        # shortcut branch
        shortcut = data

        # first block of ResNet module are 1x1 CONVs
        bm1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
        act1 = Activation('relu')(bm1)
        conv1 = Conv2D(int(K * .25), (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act1)
        
        # second block of ResNet module are 3x3 CONVs
        bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding='same', use_bias=False,
            kernel_regularizer=l2(reg))(act2)
        
        # third block of ResNet module is another set of 1x1 CONVs
        bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv2)
        act3 = Activation('relu')(bn3)
        conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

        # if reduce, using convolute on original input
        if red:
            shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, kernel_regularizer=l2(reg))(act1)

        x = add([conv3, shortcut])

        return x

    @staticmethod
    def build(width, height, depth, classes, stages, filters, reg=0.0001, bnEps=2e-5, bnMom=0.9, dataset='cifar'):
        """
        Args:
            stages: list of number of resnet module each stage needs
            filters: except the first one as init convolution layer, rest are filters of each resnet stage
        """
        inputShape = height, width, depth
        chanDim = -1
        
        if K.image_data_format() == 'channels_first':
            inputShape = depth, height, width
            chanDim = 1
        
        inputs = Input(shape=inputShape)
        # here bn acts as normalization
        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(inputs)

        if dataset == 'cifar':
            x = Conv2D(filters[0], (3, 3), use_bias=False, padding='same', kernel_regularizer=l2(reg))(x)

        for i in range(0, len(stages)):
            stride = (1, 1) if i == 0 else (2, 2)
            x = ResNet.residual_module(x, filters[i+1], stride, chanDim, red=True, bnEps=bnEps, bnMom=bnMom)
            
            for j in range(0, stages[i] - 1):
                x = ResNet.residual_module(x, filters[i+1], (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)


        x = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((8, 8))(x)
        # softmax classifier
        x = Flatten()(x)
        x = Dense(classes, kernel_regularizer=l2(reg))(x)
        x = Activation('softmax')(x)

        # create the model
        model = Model(inputs, x, name='resnet')

        return model