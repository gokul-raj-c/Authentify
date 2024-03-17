from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from keras.models import load_model
from PIL import Image
import numpy as np
from skimage import transform
from mtcnn import MTCNN
import cv2

rslt = None
file = None

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

model = load_model("model\deepfake_resnet50.h5") 
detector = MTCNN()

@app.route("/")
def index():
    return render_template("index.html") 

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=["POST","GET"])
def predict():
    global rslt, file, model, detector
    if request.method == "POST":
        class_label = ["Fake","Real"]
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(UPLOAD_FOLDER, filename))

        pt = os.path.join(os.getcwd(), UPLOAD_FOLDER, filename)

        file = filename

        image = cv2.imread(pt)
        
        faces = detector.detect_faces(image)
        
        if faces:
            face = faces[0] 
            x, y, w, h = face['box']
            face_image = image[y:y+h, x:x+w]
            face_image = cv2.resize(face_image, (128, 128))  
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_image = np.array(face_image).astype('float32') / 255
            face_image = np.expand_dims(face_image, axis=0)

            pred = model.predict(face_image)
            c = np.argmax(pred)
            rslt = class_label[c]
            return "done"
        else:
            rslt = -1
            return "no face"
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