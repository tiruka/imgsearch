from preprocess.cnn_train import CNNTrainModel
from preprocess.gen_data import LoadImage
from settings import (
    DATA_DIRECTORY,
    MODEL_PATH,
    MAPPING_JSON,
    NP_DATA,
)

if __name__ == "__main__":
    train_model = CNNTrainModel(MODEL_PATH, MAPPING_JSON, NP_DATA, batch_size=64, epochs=100)
    train_model.run()