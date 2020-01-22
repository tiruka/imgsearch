import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyControl(object):
    
    def __init__(self, np_train_data, np_test_data):
        self.np_train_data = np_train_data
        self.np_test_data = np_test_data

    def divide_save_npy(self, train, test):
        logger.info('save train data')
        x, y = train
        np.savez_compressed(self.np_train_data, x=train[0], y=train[1])
        logger.info('save test data')
        x, y = test
        np.savez_compressed(self.np_test_data, x=test[0], y=test[1])

    def divide_load_npy(self, data_type):
        if data_type == 'train':
            npz = np.load(self.np_train_data)
            X_train = npz['x']
            Y_train = npz['y']
            return X_train, Y_train
        if data_type == 'test':
            npz = np.load(self.np_train_data)
            X_test = npz['x']
            Y_test = npz['y']
            return X_test, Y_test
        raise Exception