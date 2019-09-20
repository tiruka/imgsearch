import os
import glob

from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

from preprocess.predict import PredictImage
from preprocess.gen_data import LoadImage
from settings import (
    MODEL_PATH,
    MAPPING_JSON,
    DATA_DIRECTORY,
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def ret_predicted_img_url(image_name):
    fp = os.path.join(DATA_DIRECTORY, image_name, '*.jpg')
    files = glob.glob(fp)
    return sorted(files)[0]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('File does not exist')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('File does not exist')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_name, percentage = PredictImage(Image.open(filepath), MODEL_PATH, MAPPING_JSON).print_result()
            predicted_img_url = ret_predicted_img_url(image_name)
            return render_template('result.html',
                                    original_img_url=filepath,
                                    predicted_img_url=predicted_img_url,
                                    image_name=image_name,
                                    percentage=percentage,)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)