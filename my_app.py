import os
import tensorflow as tf
graph = tf.get_default_graph()
 
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_json import FlaskJSON, JsonError, json_response, as_json
from flask_cors import CORS, cross_origin
import logging
from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import numpy as np
import argparse
import imutils
import cv2
import time
import uuid
import base64
import requests



logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('HELLO WORLD')

img_width, img_height = 150, 150
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
#model.load_weights(model_weights_path)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])


def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predict(file):
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    with graph.as_default():
        array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    return answer

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['JSON_ADD_STATUS'] = False
FlaskJSON(app)

# route for upload image
@app.route('/upload', methods=['POST'])
@as_json
def fileUpload():
    # get file
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # make prediction about image
        result = predict(file_path)
        
        # send http request to know more about the label
        payload = str(result)
        url = 'http://localhost:3002/api/landmarks/' + payload
        response = requests.get(url)
        print(response.url)
        json_data = response.json()
        print(json_data)

        # rename the file for security
        print(file_path)
        filename = my_random_string(6) + filename
        os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print(filename)
        return dict({'ob':json_data, 'dest':filename})


from flask import send_from_directory

# route for getting the uploaded image saved on the server
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})


if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(debug=True,host="localhost",use_reloader=False)




