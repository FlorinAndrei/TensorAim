from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
from pprint import pprint

# CUDA vs CPU
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def create_model():
    # build the model
    # flat 1D layer
    # dense 128-node layer
    # dense softmax output layer
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(90, 120)),
        keras.layers.Dense(1800, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    
    # compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    print(model.summary())
    return model

# how to save the trained model
checkpoint_path = os.path.normpath("training/sentry-{epoch:04d}.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq='epoch')

"""
# load train/test data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
"""

print('load data')
syndata = pickle.load(open('synthetic_data.p', 'rb'))
syntot = len(syndata)
syntrain = int(syntot * 80 / 100)

train_images = np.empty((syntrain, 90, 120), dtype=np.uint8)
train_labels = np.empty((syntrain), dtype=np.uint8)
test_images = np.empty((syntot - syntrain, 90, 120), dtype=np.uint8)
test_labels = np.empty((syntot - syntrain), dtype=np.uint8)
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
print('train/test data size:', train_images.shape, len(train_labels), test_images.shape, len(test_labels))

# normalize pixel values (0...1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# show first image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# show first 25 images, sanity check
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(train_labels[i])
plt.show()

model = create_model()

log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.normpath(log_dir), histogram_freq=1)

# train the model
model.fit(train_images, train_labels, epochs=5, callbacks = [cp_callback, tensorboard_callback])

# final save as H5
model.save('sentry.h5')

latest = tf.train.latest_checkpoint(checkpoint_dir)
print('latest:', latest)

# evaluate the model from the last save
#model = create_model()
#model.load_weights(latest)
model = keras.models.load_model('sentry.h5')
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print('Test accuracy:', test_acc)
