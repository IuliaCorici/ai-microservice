#Usage: python predict-multiclass.py
#https://github.com/tatsuyah/CNN-Image-Classifier

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 150, 150
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Label: Arcul_de_triumf")
  elif answer == 1:
	  print("Label: Ateneu")
  elif answer == 2:
	  print("Label: Palatul_parlamentului")
  return answer

  

arc_t = 0
arc_f = 0
ateneu_t = 0
ateneu_f = 0
palat_t = 0
palat_f = 0

for i, ret in enumerate(os.walk('./test-data/arcul_de_triumf')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Daisy")
    result = predict(ret[0] + '/' + filename)
    if result == 0:
      arc_t += 1
    else:
      arc_f += 1

for i, ret in enumerate(os.walk('./test-data/ateneu')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Rose")
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      ateneu_t += 1
    else:
      ateneu_f += 1

for i, ret in enumerate(os.walk('./test-data/palatul_parlamentului')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Sunflower")
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      print(ret[0] + '/' + filename)
      palat_t += 1
    else:
      palat_f += 1

"""
Check metrics
"""
print("True Daisy: ", arc_t)
print("False Daisy: ", arc_f)
print("True Rose: ", ateneu_t)
print("False Rose: ", ateneu_f)
print("True Sunflower: ", palat_t)
print("False Sunflower: ", palat_f)
