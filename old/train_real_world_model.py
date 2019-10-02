import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow import keras

import config
from model import create_model


parser = argparse.ArgumentParser(description='train a model with real world data')
parser.add_argument('--view', action='store_true', help='view training image samples')
parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
args = parser.parse_args()

datafile = 'real_world_data.h5'
h5 = h5py.File(datafile, 'r')

imshape = h5['images'].shape
lbshape = h5['labels'].shape
images = np.array(h5['images'][:])
labels = np.array(h5['labels'][:])

h5.close()

# "double" the size of the dataset by flipping images horizontally
images_flip = np.flip(images, 2)
labels_flip = labels
images_big = np.concatenate((images, images_flip), axis=0)
labels_big = np.concatenate((labels, labels_flip), axis=0)
images = images_big
labels = labels_big

del images_flip, images_big, labels_flip, labels_big

# shuffle (randomize) the data set
# because successive images might be similar
randindex = np.arange(len(labels))
np.random.shuffle(randindex)
images = images[randindex, :, :, :]
labels = labels[randindex]

lentotal = len(labels)
lentrain = int(lentotal * 80 / 100)

images_train = np.empty((lentrain, config.imgh, config.imgw, 1), dtype=np.float32)
labels_train = np.empty((lentrain), dtype=np.uint8)
images_tests = np.empty((lentotal - lentrain, config.imgh, config.imgw, 1), dtype=np.float32)
labels_tests = np.empty((lentotal - lentrain), dtype=np.uint8)

images_train = images[:lentrain, :, :, :]
labels_train = labels[:lentrain]
images_tests = images[lentrain:, :, :, :]
labels_tests = labels[lentrain:]

del images
del labels

images_train = images_train / 255.0
images_tests = images_tests / 255.0

if args.view:
  # show first 25 images, sanity check
  plt.figure(figsize=(10,10))
  for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    imsample = images_train[i].reshape((config.imgh,config.imgw,))
    plt.imshow(imsample, 'gray')
    plt.xlabel(labels_train[i])
  plt.show()

model = create_model()

model.fit(images_train, labels_train, epochs=args.epochs)

keras.models.save_model(model, 'real_world_model.h5')

model = keras.models.load_model('real_world_model.h5')
test_loss, test_acc = model.evaluate(images_tests, labels_tests, verbose=0)
print('Test accuracy:', test_acc)
