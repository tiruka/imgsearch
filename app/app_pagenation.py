import os
import glob

from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from flask_paginate import Pagination, get_page_parameter


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

img_data_dirs = [i for i in os.listdir('static/data/img') if i not in ['.gitignore', '.DS_Store']]
per_page = 3

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def ret_predicted_img_url(image_name):
    fp = os.path.join(DATA_DIRECTORY, image_name, '*.jpg')
    files = glob.glob(fp)
    return sorted(files)[0]

@app.route('/pagenation', methods=['GET', 'POST'])
def pic_view():
    if request.method == 'GET':
        page = request.args.get(get_page_parameter(), type=int, default=1)
        res = img_data_dirs[(page - 1) * per_page: page * per_page]
        name_image_set = []
        for r in res:
            img_url = glob.glob(os.path.join('static/data/img', r, '*'))
            if img_url:
                name_image_set.append((r, img_url[0]))
        pagination = Pagination(page=page, total=len(img_data_dirs),  per_page=per_page, css_framework='semantic')
        return render_template('pagenation.html', rows=name_image_set, pagination=pagination)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)