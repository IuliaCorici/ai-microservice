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

 
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def predict(file):
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    with graph.as_default():
        array = model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    return answer

def flatten_json(y): 
    out = {} 
  
    def flatten(x, name =''): 
        # If the Nested key-value  
        # pair is of dict type 
        if type(x) is dict:   
            for a in x: 
                flatten(x[a], name + a + '_') 
                  
        # If the Nested key-value 
        # pair is of list type 
        elif type(x) is list: 
            i = 0
            for a in x:                 
                flatten(a, name + str(i) + '_') 
                i += 1
        else: 
            out[name[:-1]] = x 
    flatten(y) 
    return out 

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['JSON_ADD_STATUS'] = False
FlaskJSON(app)

# route for upload image
@app.route('/upload', methods=['POST'])
def fileUpload():
    # get file
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # encode image with base64
        base64_img = get_base64_encoded_image(file_path)

        # make prediction about image
        result = predict(file_path)

        # remove the saved img
        os.remove(file_path)
        
        # send http request to know more about the label
        payload = str(result)
        url = 'http://localhost:3002/api/landmarks/' + payload
        response = requests.get(url)
        json_data = response.json()
        

        # create json with data about image and the encoded img
        my_dict = dict({'ob': json_data, "to_dec": base64_img})
        merged_dict = flatten_json(my_dict)
        
        return merged_dict


if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(debug=True,host="localhost",use_reloader=False)




