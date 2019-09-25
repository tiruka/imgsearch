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

    def __init__(self, model_name, mapping_json, np_data, batch_size=32, epochs=100,):
        self.num_label = None
        self.model_name = model_name
        self.mapping_json = mapping_json
        self.np_data = np_data
        self.batch_size = batch_size
        self.epochs = epochs

    def run(self):
        self.update_label_num()
        X_train, X_test, Y_train, Y_test = self.load_npy()

        X_train = self.normalize_data(X_train)
        X_test = self.normalize_data(X_test)
        
        Y_train = self.convert_one_hot_vector(Y_train)
        Y_test = self.convert_one_hot_vector(Y_test)

        model = self.train_model(X_train, Y_train)
        self.model_eval(model, X_test, Y_test)

    @staticmethod
    def load_index_label_mapping(fp):
        with open(fp, 'r') as f:
            index_label_mapping = json.load(f)
        return index_label_mapping

    def load_npy(self):
        X_train, X_test, Y_train, Y_test = np.load(self.np_data)
        return X_train, X_test, Y_train, Y_test

    def normalize_data(self, np_data):
        return np_data.astype(np.float128) / 255

    def convert_one_hot_vector(self, label_data):
        return np_utils.to_categorical(label_data, self.num_label)

    def update_label_num(self):
        index_label_mapping = self.load_index_label_mapping(self.mapping_json)
        self.num_label = len(index_label_mapping)

    def build_model(self, X, Y):
        model = Sequential()
        # 1st layer
        model.add(Conv2D(32, (3, 3), padding='same', 
                        input_shape=X.shape[1:]))
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
        return model

    def train_model(self, X, Y):
        model = self.build_model(X, Y)
        model.fit(X, Y,
            batch_size=self.batch_size,
            epochs=self.epochs)
        model.save(self.model_name)
        return model

    def model_eval(self, model, X, Y):
        scores = model.evaluate(X, Y, verbose=1)
        print('Test loss: {} / Test accuracy: {}'.format(scores[0], scores[1]))

