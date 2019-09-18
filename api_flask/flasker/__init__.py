from flask import Flask

app = Flask(__name__)
app.config.from_object('flasker.config')

import flasker.views