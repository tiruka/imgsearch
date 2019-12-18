import sys
import logging
from keras.models import load_model as keras_load_model
import tensorflow as tf
from PIL import Image
from preprocess.gen_data import LoadImage
from preprocess.cnn_train import CNNTrainModel
import numpy as np

logging.basicConfig(level=logging.ERROR)


class PredictImage(object):

    def __init__(self, target_image, modelpath, mapping_json):
        self.target_image = target_image
        self.modelpath = modelpath
        self.mapping_json = mapping_json

    def load_model(self):
        model = keras_load_model(self.modelpath)
        return model

    def predict(self):
        model = self.load_model()
        converted_image = LoadImage.convert_resize_image(self.target_image)
        np_data = np.asarray(converted_image) / 255
        X = np.array([np_data])
        result = model.predict(X)
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