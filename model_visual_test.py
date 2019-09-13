import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import time


parser = argparse.ArgumentParser(description='test the model')
parser.add_argument('--cli', action='store_true', help='run without GUI')
args = parser.parse_args()

model = keras.models.load_model('sentry.h5')
tr_data = pickle.load(open('synthetic_data.p', 'rb'))
img_count = len(tr_data)
test_img = np.empty((1, 90, 120, 1), dtype=np.float16)

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
  imr = obj[2].reshape((90,120,))
  plt.figure()
  plt.imshow(imr)
  plt.colorbar()
  plt.grid(False)
  legend = pred + '    ' + dur
  plt.xlabel(legend)
  if args.cli:
    print(legend)
    time.sleep(1)
  else:
    plt.show()
