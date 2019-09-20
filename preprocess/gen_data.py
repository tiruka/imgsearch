import os
import glob
import json
import itertools
import numpy as np
from sklearn import model_selection
from PIL import Image



class LoadImage(object):
    IMAGE_SIZE = 50

    def __init__(self, data_dir, np_data, mapping_json):
        self.data_dir = data_dir
        self.np_data = np_data
        self.mapping_json = mapping_json

    def run(self):
        img_dir = self.list_img_dir()
        data = []
        label = []
        index_label_mapping = {}
        for index, img_name in enumerate(img_dir):
            fp = os.path.join(self.data_dir, img_name, "*.jpg")
            images = glob.glob(fp)
            index_label_mapping.setdefault(index, img_name)
            for np_data in self.image_convert_to_np(images):
                label.append(index)
                data.append(np_data)
        data = np.array(data)
        label = np.array(label)
        self.generate_cross_validation_data(data, label)
        self.dump_index_label_mapping(index_label_mapping)

    def list_img_dir(self):
        return os.listdir(self.data_dir)

    def image_convert_to_np(self, images):
        amplified_images = []
        for fp in images:
            image = Image.open(fp)
            image = __class__.convert_resize_image(image)
            amplified_images.append(image)
            for i in itertools.chain(AmplifyData.rotate_image_data(image), AmplifyData.transpose_image_data(image)):
                amplified_images.append(i)
        for img in amplified_images:
            yield np.asarray(img)

    @classmethod
    def convert_resize_image(cls, image):
        image = image.convert('RGB')
        image = image.resize((cls.IMAGE_SIZE, cls.IMAGE_SIZE))
        return image

    def generate_cross_validation_data(self, X, Y):
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
                                            X, Y, test_size=0.33, random_state=42)
        xy = (X_train, X_test, Y_train, Y_test)
        np.save(self.np_data, xy)

    def dump_index_label_mapping(self, dic):
        with open(self.mapping_json, 'w') as f:
            json.dump(dic, f)


class AmplifyData(object):

    @classmethod
    def rotate_image_data(cls, image):
        for angle in range(-90, 90 + 1, 10):
            rotated_img = image.rotate(angle)
            yield rotated_img

    @classmethod
    def transpose_image_data(cls, image):
        yield image.transpose(Image.FLIP_LEFT_RIGHT)
        yield image.transpose(Image.FLIP_TOP_BOTTOM)
