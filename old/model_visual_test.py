import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import time

import config


parser = argparse.ArgumentParser(description='test the model')
parser.add_argument('--cli', action='store_true', help='run without GUI')
parser.add_argument('--nosleep', action='store_true', help='in CLI mode do not sleep between steps')
parser.add_argument('--noprint', action='store_true', help='max speed in CLI mode')
args = parser.parse_args()

model = keras.models.load_model('saved_model.h5')
print(model.summary())
tr_data = pickle.load(open('synthetic_data.p', 'rb'))
img_count = len(tr_data)
test_img = np.empty((1, config.imgh, config.imgw, 1), dtype=np.float16)

for i in range(0, img_count):
  obj = tr_data[i]
  maybe = obj[2]
  test_img[0, :, :, :] = obj[2] / 255.0
  tstart = time.time()
  p = model.predict(test_img)
  tend = time.time()
  dur = str(tend - tstart)
  pred = ''
  if np.argmax(p) == 0:
    pred = 'square'
  if np.argmax(p) == 1:
    pred = 'circle'
  legend = pred + '    ' + dur
  if args.cli:
    if args.noprint:
      pass
    else:
      print(legend)
    if args.nosleep:
      pass
    else:
      time.sleep(1)
  else:
    imr = obj[2].reshape((config.imgh,config.imgw,))
    fig = plt.figure()
    plt.imshow(imr)
    plt.colorbar()
    plt.grid(False)
    plt.xlabel(legend)
    plt.show()
