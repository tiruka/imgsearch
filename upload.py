import os
import glob

from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

from preprocess.predict import PredictImage
from preprocess.gen_data import LoadImage

UPLOAD_FOLDER = './uploads'
ORIGINAL_IMAGE_FOLDER = './preprocess/data/img'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def ret_predicted_img_url(image_name):
    fp = os.path.join(ORIGINAL_IMAGE_FOLDER, image_name, '*.jpg')
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
            image_name, percentage = PredictImage(Image.open(filepath)).print_result()
            predicted_img_url = ret_predicted_img_url(image_name)
            return render_template('result.html',
                                    original_img_url=filepath,
                                    predicted_img_url=predicted_img_url,
                                    percentage=percentage,)

    return render_template('index.html')

@app.route('/result')
def result(filepath):
    return render_template('result.html', img_url=filepath)

if __name__ == '__main__':
    app.run()