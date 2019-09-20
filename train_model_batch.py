from preprocess.cnn_train import CNNTrainModel
from preprocess.gen_data import LoadImage
from settings import (
    DATA_DIRECTORY,
    MODEL_PATH,
    MAPPING_JSON,
    NP_DATA,
)

if __name__ == "__main__":
    LI = LoadImage(DATA_DIRECTORY, NP_DATA, MAPPING_JSON)
    LI.run()
    train_model = CNNTrainModel(MODEL_PATH, MAPPING_JSON, NP_DATA)
    train_model.run()
