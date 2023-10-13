import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import argparse
import pygame

import config


parser = argparse.ArgumentParser(description='check the model with live images from camera')
parser.add_argument('--defdriver', action='store_true', help='use default system video driver instead of DSHOW')
parser.add_argument('--model', type=str, help='model to load')
args = parser.parse_args()

if args.defdriver:
  # use default system driver
  cap = cv2.VideoCapture(0)
else:
  # use DSHOW to get rid of the letterboxed format on some cameras
  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# use the smallest available resolution
ret = cap.set(3,640)
ret = cap.set(4,480)

model = keras.models.load_model(args.model)
print(model.summary())

pygame.mixer.init()
pygame.mixer.music.load('pew.wav')

while(True):
  t1 = int(round(time.time() * 1000))
  # capture frame
  ret, frame = cap.read()
  # grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # scale down
  fframe = cv2.resize(gray, (config.imgw, config.imgh), interpolation=cv2.INTER_AREA)
  # display
  cv2.imshow('camera', fframe)
  show2net = fframe.reshape(1, config.imgh, config.imgw, 1) / 255.0
  p = model.predict(show2net)
  t2 = int(round(time.time() * 1000))
  pred = p[0]
  if pred[1] > 0.75:
    beep = '    ############### HUMAN DETECTED! ################'
    pygame.mixer.music.play()
  else:
    beep = ''
  print(pred, '\t', int(1000/(t2-t1)), 'fps' + beep)
  # quit
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
