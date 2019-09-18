from flask import request, redirect, url_for, render_template, flash
from flasker import app
@app.route('/')
def index():
    return 'Hello world!'