from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from PIL import Image
import numpy as np
from skimage import transform

rslt = None
file = None

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

model = load_model("model/densenet.h5") 

@app.route("/")
def index():
    return render_template("index.html") 

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=["POST","GET"])
def predict():
    global rslt, file, model
    if request.method == "POST":
        class_label = ["Fake","Real"]
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(UPLOAD_FOLDER, filename))

        pt = os.path.join(os.getcwd(), UPLOAD_FOLDER, filename)

        file = filename

        image = load(pt) 
        pred = model.predict(image)
        print(pred)
        c = np.argmax(pred)
        rslt = class_label[c]
        return "res"
    else:
        return render_template("index.html")

@app.route("/result")
def result():
    global rslt, file
    if rslt==None:
        return render_template('index.html')
    return render_template('result.html',rslt=rslt, file=file)

@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (128, 128, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image 