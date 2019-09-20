import sys
from PIL import Image
from preprocess.predict import PredictImage
from settings import (
    MODEL_PATH,
    MAPPING_JSON
)

if __name__ == "__main__":
    image = Image.open(sys.argv[1])
    PI = PredictImage(image, MODEL_PATH, MAPPING_JSON)
    PI.print_result(local=True)