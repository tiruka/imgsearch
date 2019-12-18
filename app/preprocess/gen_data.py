import os
import glob
import json
import itertools
import logging
import numpy as np
from sklearn import model_selection
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadImage(object):
    IMAGE_SIZE = 300

    @classmethod
    def convert_resize_image(cls, image):
        image = image.convert('RGB')
        image = image.resize((cls.IMAGE_SIZE, cls.IMAGE_SIZE))
        return image

class ConvertImageNumpy(object):

    def __init__(self, data_dir, np_data, mapping_json):
        self.data_dir = data_dir
        self.np_data = np_data
        self.mapping_json = mapping_json
        self.img_extension = "*.JPG"

    def run(self):
        img_dir = self.list_img_dir()
        data = []
        label = []
        index_label_mapping = {}
        for index, img_name in enumerate(img_dir):
            if index % 1000 == 0:
                logger.info(f'index:{index}\t{img_name}')
            fp = os.path.join(self.data_dir, img_name, self.img_extension)
            images = glob.glob(fp)
            index_label_mapping.setdefault(index, img_name)
            for np_data in self.image_convert_to_np(images):
                label.append(index)
                data.append(np_data)
        logger.info('Transform data into numpy array')
        data = np.array(data)
        label = np.array(label)
        self.generate_cross_validation_data(data, label)
        self.dump_index_label_mapping(index_label_mapping)

    def list_img_dir(self):
        return os.listdir(self.data_dir)

    def image_convert_to_np(self, images):
        for fp in images:
            yield np.asarray(Image.open(fp))

    def generate_cross_validation_data(self, X, Y):
        logger.info('generate_cross_validation_data')
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
                                            X, Y, test_size=0.33, random_state=42)
        xy = (X_train, X_test, Y_train, Y_test)
        logger.info('save_cross_validation_data')
        np.save(self.np_data, xy)

    def dump_index_label_mapping(self, dic):
        logger.info('index_label_mapping.json creating')
        with open(self.mapping_json, 'w') as f:
            json.dump(dic, f)


class AugmentImage(object):

    def __init__(self, data_dir, img_extension=None):
        self.data_dir = data_dir
        self.img_extension = img_extension or ".JPG"

    def run(self):
        img_dir = self.list_img_dir()
        for index, img_name in enumerate(img_dir):
            logger.debug(f'index:{index}\t{img_name}')
            fp = os.path.join(self.data_dir, img_name, "*" + self.img_extension)
            images = glob.glob(fp)
            self.augment_save_images(images)

    def list_img_dir(self):
        return os.listdir(self.data_dir)

    def augment_save_images(self, images):
        '''
        ToDo
        resize -> augment or augment -> resize
        which would be better?
        '''
        for fp in images:
            image = Image.open(fp)
            name = fp.rstrip(self.img_extension)
            for i, augmented_image in enumerate(itertools.chain(
                                AugmentImage.rotate_image_data(image),
                                AugmentImage.transpose_image_data(image))):
                augmented_image = LoadImage.convert_resize_image(augmented_image)
                path = f'{name}_{i}_{self.img_extension}'
                augmented_image.save(path, qualit=95)
            image = LoadImage.convert_resize_image(image)
            os.remove(fp)
            image.save(fp, quality=95)

    @classmethod
    def rotate_image_data(cls, image):
        for angle in range(-180, 180 + 1, 10):
            rotated_img = image.rotate(angle)
            yield rotated_img

    @classmethod
    def transpose_image_data(cls, image):
        yield image.transpose(Image.FLIP_LEFT_RIGHT)
        yield image.transpose(Image.FLIP_TOP_BOTTOM)
