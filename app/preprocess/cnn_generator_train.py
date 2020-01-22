from datetime import date, datetime
import glob
import json
import pickle
import logging
import math
import os

from keras import optimizers
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Dropout,
    Flatten,
    Dense
)
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneratorCNNModel(object):

    def __init__(self, IMG_DIR, MODEL_DIR, CHECK_POINT, LOG_DIR, AUGMENTED_IMG_DIR=None, batch_size=64, epochs=10, resume=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.IMG_DIR = IMG_DIR
        self.MODEL_DIR = MODEL_DIR
        self.CHECK_POINT = CHECK_POINT
        self.LOG_DIR = LOG_DIR
        self.AUGMENTED_IMG_DIR = AUGMENTED_IMG_DIR
        self.input_shape = (300, 300, 3)
        self.img_train_gen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=90,
            width_shift_range=1,
            height_shift_range=2,
            brightness_range=(0.5, 1),
            shear_range=.8,
            zoom_range=.2,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=preprocess_input)
        self.img_test_gen = ImageDataGenerator(rescale=1./255)
        self.img_train_iters = self.img_validation_iters = None
        self.steps_per_epoch = self.validation_steps = None
        self.multiple = 10
        self.set_generator()
        self.stamp = datetime.now().strftime('%Y%m%d%H%M%S')
        self.resume = resume

    def run(self):
        self.pre_calculation()
        self.train_generator_model()

    def set_generator(self):
        self.img_train_iters = self.create_img_iters('train', self.IMG_DIR)
        self.img_validation_iters = self.create_img_iters('test', self.IMG_DIR) #一旦、同じトレインと検証は同じ場所

    def pre_calculation(self):
        self.steps_per_epoch = self.cal_steps_for_epoch(self.img_train_iters)
        self.validation_steps = self.cal_steps_for_epoch(self.img_validation_iters)

    def create_img_iters(self, kind, directory):
        iters_kwargs = dict(
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            seed=42,
        )
        if kind == 'train':
            if self.AUGMENTED_IMG_DIR:
                iters_kwargs.update(
                    save_format='png',
                    save_prefix='augmented',
                    save_to_dir=self.AUGMENTED_IMG_DIR)
            img_iters = self.img_train_gen.flow_from_directory(
                directory,
                **iters_kwargs)
        else:
            img_iters = self.img_test_gen.flow_from_directory(
                directory,
                **iters_kwargs,)
        return img_iters

    def cal_steps_for_epoch(self, iters):
        return math.ceil(iters.samples / self.batch_size)

    def build_model(self):
        vgg16 = VGG16(include_top=False, input_shape=self.input_shape)
        model = Sequential(vgg16.layers)
        for layer in model.layers:
            layer.trainable = False

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.img_train_iters.num_classes))
        model.add(Activation('softmax'))

        opt = optimizers.SGD(lr=1e-4, momentum=0.9)
        model.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
        if self.resume:
            model.load_weights(self._search_latest_weight_model(self.resume))
        return model

    def _search_latest_weight_model(self, name):
        if name == 'model':
            weights = glob.glob(os.path.join(self.MODEL_DIR, '*.h5'))
        elif name == 'checkpoint':
            weights = glob.glob(os.path.join(self.CHECK_POINT, '*.h5'))
        if len(weights) < 1:
            raise Exception('No model exists')
        weights.sort(reverse=True)
        logger.info('Load model from {}'.format(weights[0]))
        return weights[0]

    def train_generator_model(self):
        model = self.build_model()
        if self.resume is None:
            self.save_label_index()
            self.save_network(model)
        cp = self._add_check_point()
        csv = self._add_logging()
        history = model.fit_generator(
            self.img_train_iters,
            steps_per_epoch=self.steps_per_epoch * self.multiple,
            epochs=self.epochs,
            validation_data=self.img_validation_iters,
            validation_steps=self.validation_steps,
            callbacks=[cp, csv]
        )
        model.save_weights(os.path.join(
            self.MODEL_DIR, '{}_cnn_gen_final.h5'.format(self.stamp)))
        return model

    def save_network(self, model):
        model_json = os.path.join(self.MODEL_DIR, 'model.json')
        with open(model_json, 'w') as f:
            json.dump(model.to_json(), f)
            
    def save_label_index(self):
        # save pairs of labels and indices
        model_classes = os.path.join(self.MODEL_DIR, 'classes.pkl')
        with open(model_classes, 'wb') as f:
            pickle.dump(self.img_train_iters.class_indices, f)

    def _add_check_point(self):
        cp_filepath = os.path.join(
            self.CHECK_POINT, self.stamp + '_ep_{epoch:02d}_ls_{loss:.1f}.h5')
        cp = ModelCheckpoint(
            cp_filepath,
            monitor='loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            period=5
        )
        return cp

    def _add_logging(self):
        csv_filepath = os.path.join(self.LOG_DIR, '{}_loss.csv'.format(self.stamp))
        csv = CSVLogger(csv_filepath, append=True)
        return csv