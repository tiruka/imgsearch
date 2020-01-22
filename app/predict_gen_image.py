import sys
from PIL import Image
from preprocess.predict import PredictVGG16Image
from settings import (
    MODEL_DIR,
)

if __name__ == "__main__":
    PI = PredictVGG16Image(MODEL_DIR)
    if sys.argv[1] == 'save_results':
        PI.save_results_json()
    else:
        target_image_path = sys.argv[1]
        PI.print_results(target_image_path, num=10, local=True)