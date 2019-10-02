from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import argparse
import os

import config
from model import create_model


parser = argparse.ArgumentParser(description='train the model')
parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--view', action='store_true', help='view training image samples')
parser.add_argument('--cpu', action='store_true', help='run on CPU instead of GPU')
args = parser.parse_args()

if args.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# how to save the trained model
checkpoint_path = os.path.normpath("training/sentry-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

print('load data')
syndata = pickle.load(open('synthetic_data.p', 'rb'))
syntot = len(syndata)
syntrain = int(syntot * 80 / 100)

train_images = np.empty((syntrain, config.imgh, config.imgw, 1), dtype=np.float32)
train_labels = np.empty((syntrain, 2), dtype=np.float32)
test_images = np.empty((syntot - syntrain, config.imgh, config.imgw, 1), dtype=np.float32)
test_labels = np.empty((syntot - syntrain, 2), dtype=np.float32)
for i in range(0, syntrain):
    obj = syndata[i]
    train_images[i, :, :] = obj[2]
    train_labels[i] = obj[0]
for i in range(syntrain, syntot):
    j = i - syntrain
    obj = syndata[i]
    test_images[j, :, :] = obj[2]
    test_labels[j] = obj[0]

# print shape/size for train/test data
print('\n', 'train/test data size:', train_images.shape, len(train_labels), test_images.shape, len(test_labels), '\n')

# normalize pixel values (0...1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# show first image
plt.figure()
imsample = train_images[0].reshape((config.imgh,config.imgw,))
plt.imshow(imsample)
plt.colorbar()
plt.grid(False)
if args.view:
    plt.show()

# show first 25 images, sanity check
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    imsample = train_images[i].reshape((config.imgh,config.imgw,))
    plt.imshow(imsample)
    plt.xlabel(train_labels[i])

if args.view:
    plt.show()

model = create_model()

log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.normpath(log_dir), histogram_freq=1)

# train the model
model.fit(train_images,
          train_labels,
          epochs=args.epochs)
#          callbacks = [cp_callback, tensorboard_callback])

# final save
keras.models.save_model(model, 'saved_model.h5')

latest = tf.train.latest_checkpoint(checkpoint_dir)
print('latest:', latest)

# evaluate the model from the last save
model = keras.models.load_model('saved_model.h5')
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('Test accuracy:', test_acc)
