from preprocess.gen_data import ConvertImageNumpy
from settings import (
    DATA_DIRECTORY,
    MODEL_PATH,
    MAPPING_JSON,
    NP_DATA,
)

if __name__ == "__main__":
    ConvertImageNumpy(DATA_DIRECTORY, NP_DATA, MAPPING_JSON).run()