from __future__ import division, print_function
import tensorflow as tf
import tensorflow as tf
import sys
import os
import glob
import re
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

Sample_Path = 'C:/Python/Python39/Sample_Signatures_BOB.h5'

# Load your trained model
model = load_model(Sample_Path)


def model_predict(Image_Path, Sample1):
    print(Image_Path)

    Images = image.load_img(Image_Path, target_size=(520, 520))
    # Preprocessing
    var = image.img_to_array(Images)

    var = var / 255
    var = np.expand_dims(var, axis=0)

    predictions = Sample1.predict(var)
    predictions = np.argmax(predictions, axis=1)
    if predictions == 0:
        predictions = "SORRY, THE GIVEN SIGNATURE IS PREDICTED AS FRAUD"
    else:
        predictions = "SIGNATURE IS ORIGINAL"

    return predictions


@app.route('/', methods=['GET'])
def index():

    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']
        bases = os.path.dirname(__file__)
        file_path = os.path.join(
            bases, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        predictions = model_predict(Image_Path, Sample1)
        result = predictions
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001, debug=True)