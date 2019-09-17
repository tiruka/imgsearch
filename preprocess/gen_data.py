import os
import glob
import json
import numpy as np
from sklearn import model_selection
from PIL import Image

IMAGE_SIZE = 50
DATA_DIR = 'data/img'
img_dir = os.listdir(DATA_DIR)
r = img_dir[0]


class LoadImage(object):

    def __init__(self):
        self.DATA_DIR = 'data/img'
        self.IMAGE_SIZE = 50

    def run(self):
        img_dir = self.list_img_dir()
        data = []
        label = []
        index_label_mapping = {}
        for index, img_name in enumerate(img_dir):
            fp = os.path.join(self.DATA_DIR, img_name, "*.jpg")
            images = glob.glob(fp)
            index_label_mapping.setdefault(index, img_name)
            for np_data in self.image_convert_to_np(images):
                label.append(index)
                data.append(np_data)
        data = np.array(data)
        label = np.array(label)
        self.generate_cross_validation_data(data, label)
        with open('./index_label_mapping.json', 'w') as f:
            json.dump(index_label_mapping, f)

    def list_img_dir(self):
        return os.listdir(self.DATA_DIR)

    def image_convert_to_np(self, image_fp):
        for fp in image_fp:
            image = Image.open(fp)
            image = image.convert('RGB')
            image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            yield np.asarray(image)

    def generate_cross_validation_data(self, X, Y):
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
                                            X, Y, test_size=0.33, random_state=42)
        xy = (X_train, X_test, Y_train, Y_test)
        np.save('./np_data.npy', xy)

LI = LoadImage()
LI.run()