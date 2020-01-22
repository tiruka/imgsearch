import sys
from utils.utils import clear_under_vars 
from preprocess.cnn_generator_train import GeneratorCNNModel
from settings import (
    IMG_DIR,
    MODEL_DIR,
    CHECK_POINT,
    LOG_DIR,
    AUGMENTED_IMG_DIR
)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception('No Args Error')
    if sys.argv[1] == 'clear':
        # Clear All data under var directory and start learning
        clear_under_vars()
        GeneratorCNNModel(IMG_DIR, MODEL_DIR, CHECK_POINT, LOG_DIR, batch_size=64, epochs=50).run()
    elif sys.argv[1] == 'model':
        # Resume learning from the latest model under var model dir
        GeneratorCNNModel(IMG_DIR, MODEL_DIR, CHECK_POINT, LOG_DIR, batch_size=64, epochs=50, resume='model').run()
    elif sys.argv[1] == 'checkpoint':
        # Resume learning from the latest model under var checkpoint dir
        GeneratorCNNModel(IMG_DIR, MODEL_DIR, CHECK_POINT, LOG_DIR, batch_size=64, epochs=50, resume='checkpoint').run()
    elif sys.argv[1] == 'only_clearup':
        clear_under_vars()
    else:
        raise Exception('No Such Args Error')