import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

from keras.models import Sequential, load_model
import keras,sys
import numpy as np
from PIL import Image

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

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
            return render_template('result.html', original_img_url=filepath)
            # model = load_model('./cnn_aug.h5')

            # image = Image.open(filepath)
            # image = image.convert('RGB')
            # image = image.resize((image_size, image_size))
            # data = np.asarray(image)
            # X = []
            # X.append(data)
            # X = np.array(X)

            # result = model.predict([X])[0]
            # predicted = result.argmax()
            # percentage = int(result[predicted] * 100)
            # return "Label:  " + classes[predicted] + ", Probability"+ str(percentage) + " %"

    return render_template('index.html')
            # return redirect(url_for('uploaded_file', filename=filename))

@app.route('/result')
def result(filepath):
    return render_template('result.html', img_url=filepath)

if __name__ == '__main__':
    app.run()