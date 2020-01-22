import json
import logging

from keras.utils import np_utils
import numpy as np

from preprocess.cnn_model import (
    CNNCifar10,
    CNNVGG16,
)
from preprocess.numpycontrol import NumpyControl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNTrainModel(CNNVGG16):

    def __init__(self, model_name, mapping_json, np_train_data, np_test_data, batch_size=64, epochs=100,):
        self.num_label = None
        self.model_name = model_name
        self.mapping_json = mapping_json
        self.npc = NumpyControl(np_train_data, np_test_data)
        self.batch_size = batch_size
        self.epochs = epochs

    def run(self):
        self.update_label_num()
        logger.info('Load Numpy Data')
        X_train, Y_train = self.npc.divide_load_npy('train')

        logger.info('Normalizing Data for train')
        X_train = self.normalize_data(X_train)
        logger.info('Convert to One Hot Vector for train')
        Y_train = self.convert_one_hot_vector(Y_train)
        logger.info('Model Training')
        model = self.train_model(X_train, Y_train)
        del X_train, Y_train
        
        X_test, Y_test = self.npc.divide_load_npy('test')
        logger.info('Normalizing Data for test')
        X_test = self.normalize_data(X_test)
        logger.info('Convert to One Hot Vector for test')
        Y_test = self.convert_one_hot_vector(Y_test)

        logger.info('Model Evaluation')
        self.model_eval(model, X_test, Y_test)

    @staticmethod
    def load_index_label_mapping(fp):
        with open(fp, 'r') as f:
            index_label_mapping = json.load(f)
        return index_label_mapping

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
        logger.info('Test loss: {} / Test accuracy: {}'.format(scores[0], scores[1]))
