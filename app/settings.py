import os

BASE_DIR = os.path.realpath(os.path.join(os.path.dirname(__name__)))
IMG_DIR = os.path.join(BASE_DIR, 'static/data/img')
LOCAL_IMG_DIR = 'static/data/img'
VAR_DIR = os.path.join(BASE_DIR, 'var')
MODEL_PATH = os.path.join(BASE_DIR, 'img_keras_cnn.h5')
MAPPING_JSON = os.path.join(BASE_DIR, 'index_label_mapping.json')
NP_TRAIN_DATA = os.path.join(BASE_DIR, 'np_train_data.npz')
NP_TEST_DATA = os.path.join(BASE_DIR, 'np_test_data.npz')
MODEL_DIR = os.path.join(VAR_DIR, 'model')
CHECK_POINT = os.path.join(VAR_DIR, 'checkpoint')
LOG_DIR = os.path.join(VAR_DIR, 'learning_log')
RECOMMEND_RESULTS_FP = os.path.join(VAR_DIR, 'results', 'recommends.json')
AUGMENTED_IMG_DIR = os.path.join(VAR_DIR, 'augmented_images')
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'JPG'])