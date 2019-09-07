from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
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
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
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

# load train/test data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# print shape/size for train/test data
print(train_images.shape, len(train_labels), test_images.shape, len(test_labels))

# show first image
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
#plt.show()

# normalize pixel values (0...1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# show first 25 images, sanity check
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#plt.show()

model = create_model()

# train the model
model.fit(train_images, train_labels, epochs=5, callbacks = [cp_callback])

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
