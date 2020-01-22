import base64
import json
import os
import glob
import uuid

from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask_paginate import Pagination, get_page_parameter

from preprocess.predict import PredictVGG16Image
from settings import (
    MODEL_DIR,
    LOCAL_IMG_DIR,
    UPLOAD_FOLDER,
    ALLOWED_EXTENSIONS,
    RECOMMEND_RESULTS_FP
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

img_data_dirs = [i for i in os.listdir('static/data/img') if i not in ['.gitignore', '.DS_Store']]
per_page = 12
PI = PredictVGG16Image(MODEL_DIR)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def ret_img_url(image_name):
    fp = os.path.join(LOCAL_IMG_DIR, image_name, '*.png')
    files = glob.glob(fp)
    return sorted(files)[0]

def load_results_json():
    with open(RECOMMEND_RESULTS_FP, 'r') as f:
        results = json.load(f)
    return results

results = load_results_json()

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
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()) + filename)
            file.save(filepath)
            with open(filepath, 'rb') as f:
                data = f.read()
            base64_img = 'data:image/png;base64,' + base64.b64encode(data).decode('utf-8')
            try:
                image_name_list, percentages = PI.print_results(filepath, num=12, threshold=2)
            finally:
                os.remove(filepath)
            predicted_img_url_list = [ret_img_url(i) for i in image_name_list]
            predicted_img_data_set = [tupled_data for tupled_data in zip(predicted_img_url_list, image_name_list, percentages)]
            return render_template('result.html',
                                    original_img_url=base64_img,
                                    predicted_img_data_set=predicted_img_data_set,)
    return render_template('index.html')

@app.route('/pagenation', methods=['GET', 'POST'])
def pic_view():
    if request.method == 'GET':
        page = request.args.get(get_page_parameter(), type=int, default=1)
        res = img_data_dirs[(page - 1) * per_page: page * per_page]
        name_image_set = []
        for name in res:
            img_url = ret_img_url(name)
            if img_url:
                recommends = [(target, ret_img_url(target), acc) for target, acc in results.get(name, []) if acc > 3 and name != target]
                name_image_set.append((name, img_url, recommends))
        pagination = Pagination(page=page, total=len(img_data_dirs),  per_page=per_page, css_framework='semantic')
        return render_template('pagenation.html', rows=name_image_set, pagination=pagination)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4321)