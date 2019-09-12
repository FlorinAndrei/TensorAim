import tensorflow as tf
from tensorflow import keras
import numpy as np
import pickle
import matplotlib.pyplot as plt
import time


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
  plt.xlabel(pred + '    ' + dur)
  plt.show()
