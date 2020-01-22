from abc import ABC, abstractmethod

from keras import optimizers
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Dropout,
    Flatten,
    Dense
)
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19

class CNNBaseModel(ABC):
    
    @abstractmethod
    def build_model(self, X, Y):
        '''
        この中でkerasによりmodelを決定する。
        '''
        pass


class CNNCifar10(CNNBaseModel):
    '''
    CIFAR10のモデル
    '''
    def build_model(self, X, Y):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', 
                        input_shape=X.shape[1:]))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_label))
        model.add(Activation('softmax'))

        opt = optimizers.RMSprop(lr=1e-5, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
        return model

    def normalize_data(self, np_data):
        return np_data.astype(np.float128) / 255

class CNNVGG16(CNNBaseModel):
    '''
    VGG16のモデル
    '''
    def build_model(self, X, Y):
        vgg16 = VGG16(include_top=False, input_shape=X.shape[1:])
        model = Sequential(vgg16.layers)
        for layer in model.layers:
            layer.trainable = False

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_label))
        model.add(Activation('softmax'))

        # opt = optimizers.RMSprop(lr=1e-4, decay=1e-6)
        opt = optimizers.SGD(lr=1e-4, momentum=0.9)
        model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
        return model

    def normalize_data(self, np_data):
        return preprocess_input(np_data.astype(np.float128))