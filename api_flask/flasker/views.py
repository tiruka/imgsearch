from flasker import app
import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = frozenset(['png', 'jpg', 'jpeg',])
NOEXIST_MESSAGE = 'File does not exist'
NOT_ALLOWED_FILE_MESSAGE = 'Only {} are allowed'.format(', '.join([f for f in ALLOWED_EXTENSIONS]))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return NOEXIST_MESSAGE
        file = request.files['file']
        if file.filename == '':
            return NOEXIST_MESSAGE
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'] , filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
        elif not allowed_file(file.filename):
            return NOT_ALLOWED_FILE_MESSAGE
    return '''
    <!doctype html>
    <head>
    <meta charset="UTF-8">
    <title>Upload new File</title>
    </head>
    <body>
    <h1>Upload File For Image Search!</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    </body>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)