from preprocess.gen_data import AugmentImage
from settings import (
    DATA_DIRECTORY,
    MODEL_PATH,
    MAPPING_JSON,
    NP_DATA,
)

if __name__ == "__main__":
    augmentcls = AugmentImage(DATA_DIRECTORY)
    augmentcls.run()
