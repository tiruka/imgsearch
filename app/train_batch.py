import sys

from preprocess.gen_data import AugmentImage
from preprocess.gen_data import ConvertImageNumpy
from preprocess.cnn_train import CNNTrainModel
from preprocess.gen_data import LoadImage
from settings import (
    IMG_DIR,
    MODEL_PATH,
    MAPPING_JSON,
    NP_TRAIN_DATA,
    NP_TEST_DATA
)

augment = convert = train = False
if __name__ == "__main__":
    if sys.argv[1] == 'all':
        augment = convert = train = True
    elif sys.argv[1] == 'augment':
        augment = True
    elif sys.argv[1] == 'convert':
        convert = True
    elif sys.argv[1] == 'train':
        train = True

    if augment:
        print('augment')
        AugmentImage(IMG_DIR).run()
    if convert:
        print('covert')
        ConvertImageNumpy(IMG_DIR, NP_TRAIN_DATA, NP_TEST_DATA, MAPPING_JSON).run()
    if train:
        print('train')
        CNNTrainModel(MODEL_PATH, MAPPING_JSON, NP_TRAIN_DATA, NP_TEST_DATA, batch_size=64, epochs=100).run()