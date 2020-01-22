import glob
import json
import sys
import logging
import os
import pickle
from keras.models import load_model as keras_load_model
from keras import backend as keras_backend
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
from preprocess.gen_data import LoadImage
from preprocess.cnn_train import CNNTrainModel
from preprocess.cnn_generator_train import GeneratorCNNModel
import numpy as np

from settings import (
    IMG_DIR,
    MODEL_DIR,
    CHECK_POINT,
    LOG_DIR,
    RECOMMEND_RESULTS_FP
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
graph = tf.get_default_graph()

class PredictImage(object):

    def __init__(self, target_image, modelpath, mapping_json):
        self.target_image = target_image
        self.modelpath = modelpath
        self.mapping_json = mapping_json

    def load_model(self):
        keras_backend.clear_session()
        model = keras_load_model(self.modelpath)
        return model

    def predict(self):
        converted_image = LoadImage.convert_resize_image(self.target_image)
        np_data = np.asarray(converted_image) / 255
        X = np.array([np_data])
        model = self.load_model()
        result = model.predict(X)
        keras_backend.clear_session()
        return result

    def print_result(self, local=False):
        index_label_mapping = CNNTrainModel.load_index_label_mapping(self.mapping_json)
        result = self.predict()[0]
        predicted_index = result.argmax()
        percentage = float(result[predicted_index] * 100)
        image_name = index_label_mapping.get(str(predicted_index))
        if local:
            print("Image Name: {0} Accuracy: {1} %".format(image_name, percentage))
        return image_name, percentage

    def print_results(self, num=1, local=False):
        index_label_mapping = CNNTrainModel.load_index_label_mapping(self.mapping_json)
        result = self.predict()[0]
        predicted_index_list = result.argsort()[-num:][::-1]
        percentages = [float(result[i] * 100) for i in predicted_index_list]
        image_name_list = [index_label_mapping.get(str(i)) for i in predicted_index_list]
        return image_name_list, percentages

    def save_results_json(self, num=1, fp=None):
        pass


class PredictVGG16Image(PredictImage):
    def __init__(self, MODEL_DIR):
        self.MODEL_DIR = MODEL_DIR
        self.model = self.load_model()
        self.index_label_mapping = self.load_index_label_mapping()

    def load_image(self, target_image):
        img_np = img_to_array(load_img(target_image, target_size=(300, 300)))
        x = np.expand_dims(img_np, axis=0) / 255
        # x = preprocess_input(x)
        return x

    def load_model(self):
        model = GeneratorCNNModel(IMG_DIR, MODEL_DIR, CHECK_POINT, LOG_DIR,).build_model()
        files = glob.glob(os.path.join(self.MODEL_DIR, '*.h5'))
        latest = sorted(files, reverse=True)[0]
        logger.info(latest)
        model.load_weights(latest)
        return model

    def load_index_label_mapping(self):
        with open(os.path.join(self.MODEL_DIR, 'classes.pkl'), 'rb') as f:
            index_label_mapping = pickle.load(f)
        index_label_mapping = self.get_swap_dict(index_label_mapping)
        return index_label_mapping

    def predict(self, target_image):
        X = self.load_image(target_image)
        # model = self.load_model()
        # result = model.predict(X)
        with graph.as_default():
            result = self.model.predict(X)
        return result

    def print_result(self, target_image, local=False):
        result = self.predict(target_image)[0]
        predicted_index = result.argmax()
        percentage = float(result[predicted_index] * 100)
        image_name = self.index_label_mapping.get(predicted_index)
        if local:
            print("Image Name: {0} Accuracy: {1} %".format(image_name, percentage))
        return image_name, percentage

    def print_results(self, target_image, num=1, threshold=0, local=False):
        result = self.predict(target_image)[0]
        predicted_index_list = result.argsort()[-num:][::-1]
        canditates = [self.index_label_mapping.get(i) for i in predicted_index_list]
        acc = [float(result[i] * 100) for i in predicted_index_list]
        if local:
            for image_name, percentage in zip(image_name_list, percentages):
                print("Image Name: {0} Accuracy: {1} %".format(image_name, percentage))
        image_name_list = []
        percentages = []
        for i, j in zip(canditates, acc):
            if j > threshold:
                image_name_list.append(i)
                percentages.append(j)
        return image_name_list, percentages

    def get_swap_dict(self, d):
        return {v: k for k, v in d.items()}

    def save_results_json(self, num=10):
        '''
        学習用に作成したファイルをinputとして分類させ、確率を用いることで擬似的に類似画像をレコメンドする
        predict_gen_imageのみ対応
        '''
        d = dict()
        img_data_dirs = [i for i in os.listdir(IMG_DIR) if i not in ['.gitignore', '.DS_Store']]
        for i, name in enumerate(img_data_dirs):
            target_image = self.ret_predicted_img_url(name)
            image_name_list, percentages = self.print_results(target_image, num=10)
            r = [[image_name, percentage] for image_name, percentage in zip(image_name_list, percentages)]
            d[name] = r
            if i % 100 == 0:
                logger.info('{} / {} done'.format(i, len(img_data_dirs)))

        with open(RECOMMEND_RESULTS_FP, 'w') as json_fp:
            json.dump(d, json_fp, indent=2)

    def ret_predicted_img_url(self, image_name):
        fp = os.path.join(IMG_DIR, image_name, '*.png')
        files = glob.glob(fp)
        return sorted(files)[0]