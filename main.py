from preprocess.cnn_train import CNNTrainModel
from preprocess.gen_data import LoadImage

if __name__ == "__main__":
    LI = LoadImage()
    LI.run()
    train_model = CNNTrainModel()
    train_model.run()