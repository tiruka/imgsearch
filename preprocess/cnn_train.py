import json
import keras
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Dropout,
    Flatten,
    Dense
)
from keras.utils import np_utils
import numpy as np

class CNNTrainModel(object):

    def __init__(self, batch_size=32, epochs=100):
        self.num_label = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_name = './img_keras_cnn.h5'

    def run(self):
        index_label_mapping = self.load_index_label_mapping()
        X_train, X_test, Y_train, Y_test = self.load_npy()
        X_train = self.normalize_data(X_train)
        X_test = self.normalize_data(X_test)
        
        self.num_label = len(index_label_mapping)
        Y_train = self.convert_one_hot_vector(Y_train)
        Y_test = self.convert_one_hot_vector(Y_test)

        model = self.model_train(X_train, Y_train)
        self.model_eval(model, X_test, Y_test)


    def load_index_label_mapping(self):
        with open('./index_label_mapping.json', 'r') as f:
            index_label_mapping = json.load(f)
        return index_label_mapping

    def load_npy(self):
        X_train, X_test, Y_train, Y_test = np.load('./np_data.npy')
        return X_train, X_test, Y_train, Y_test

    def normalize_data(self, np_data):
        return np_data.astype(np.float128) / 255

    def convert_one_hot_vector(self, label_data):
        return np_utils.to_categorical(label_data, self.num_label)

    def model_train(self, X_train, Y_train):
        model = Sequential()
        # 1st layer
        model.add(Conv2D(32, (3, 3), padding='same', 
                        input_shape=X_train.shape[1:]))
        model.add(Activation('relu'))
        # 2nd layer
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 3rd layer
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

        opt = keras.optimizers.RMSprop(lr=1e-5, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
        model.fit(X_train, Y_train,
            batch_size=self.batch_size,
            epochs=self.epochs)
        model.save(self.model_name)
        return model

    def model_eval(self, model, X, Y):
        scores = model.evaluate(X, Y, verbose=1)
        print('Test loss: {} / Test accuracy: {}'.format(scores[0], scores[1]))

train_model = CNNTrainModel()
train_model.run()