import json
import logging

from keras.utils import np_utils
import numpy as np

from preprocess.cnn_model import (
    CNNCifar10,
    CNNVGG16,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNTrainModel(CNNVGG16):

    def __init__(self, model_name, mapping_json, np_data, batch_size=64, epochs=100,):
        self.num_label = None
        self.model_name = model_name
        self.mapping_json = mapping_json
        self.np_data = np_data
        self.batch_size = batch_size
        self.epochs = epochs

    def run(self):
        self.update_label_num()
        logger.info('Load Numpy Data')
        X_train, X_test, Y_train, Y_test = self.load_npy()

        logger.info('Normalizing Data')
        X_train = self.normalize_data(X_train)
        X_test = self.normalize_data(X_test)
        
        logger.info('Convert to One Hot Vector')
        Y_train = self.convert_one_hot_vector(Y_train)
        Y_test = self.convert_one_hot_vector(Y_test)

        logger.info('Model Training')
        model = self.train_model(X_train, Y_train)
        logger.info('Model Evaluation')
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

