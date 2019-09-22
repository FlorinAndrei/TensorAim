import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
from tensorflow import keras
from pprint import pprint


def create_model():
    
    # build the model
    model = keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(90,120,1)),
        keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(units=100, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=2, activation='softmax')
    ])
    
    # compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        
    """
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['acc'])
    """
    
    print(model.summary())
    
    return model


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

randindex = np.arange(len(labels))
np.random.shuffle(randindex)
images = images[randindex, :, :, :]
labels = labels[randindex]

lentotal = len(labels)
lentrain = int(lentotal * 80 / 100)

images_train = np.empty((lentrain, 90, 120, 1), dtype=np.float32)
labels_train = np.empty((lentrain), dtype=np.uint8)
images_tests = np.empty((lentotal - lentrain, 90, 120, 1), dtype=np.float32)
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
    imsample = images_train[i].reshape((90,120,))
    plt.imshow(imsample, 'gray')
    plt.xlabel(labels_train[i])
  plt.show()

model = create_model()

model.fit(images_train, labels_train, epochs=args.epochs)

keras.models.save_model(model, 'real_world_model.h5')

model = keras.models.load_model('real_world_model.h5')
test_loss, test_acc = model.evaluate(images_tests, labels_tests, verbose=0)
print('Test accuracy:', test_acc)
