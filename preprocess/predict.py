import sys
from keras.models import load_model as keras_load_model
from PIL import Image
from preprocess.gen_data import LoadImage
from preprocess.cnn_train import CNNTrainModel
import numpy as np


class PredictImage(object):

    def __init__(self, target_image):
        self.target_image = target_image

    def load_model(self):
        return keras_load_model('./img_keras_cnn.h5')

    def predict(self):
        model = self.load_model()
        converted_image = LoadImage.convert_resize_image(self.target_image)
        np_data = np.asarray(converted_image) / 255
        X = np.array([np_data])
        result = model.predict(X)
        return result

    def print_result(self):
        index_label_mapping = CNNTrainModel.load_index_label_mapping()
        result = self.predict()[0]
        predicted_index = result.argmax()
        percentage = int(result[predicted_index] * 100)
        image_name = index_label_mapping.get(str(predicted_index))
        print("Image Name: {0} Accuracy: {1} %".format(image_name, percentage))
        return image_name, percentage

if __name__ == "__main__":
    image = Image.open(sys.argv[1])
    PI = PredictImage(image)
    PI.print_result()
