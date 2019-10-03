# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib import colors as mcolors
import argparse
import cv2
import os
import time
import maestro
import pygame
from pprint import pprint

# fix issue: "Could not create cudnn handle"
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class BoundBox:
  def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
    self.xmin = xmin
    self.ymin = ymin
    self.xmax = xmax
    self.ymax = ymax
    self.objness = objness
    self.classes = classes
    self.label = -1
    self.score = -1

  def get_label(self):
    if self.label == -1:
      self.label = np.argmax(self.classes)

    return self.label

  def get_score(self):
    if self.score == -1:
      self.score = self.classes[self.get_label()]

    return self.score

def _sigmoid(x):
  return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
  grid_h, grid_w = netout.shape[:2]
  nb_box = 3
  netout = netout.reshape((grid_h, grid_w, nb_box, -1))
  nb_class = netout.shape[-1] - 5
  boxes = []
  netout[..., :2]  = _sigmoid(netout[..., :2])
  netout[..., 4:]  = _sigmoid(netout[..., 4:])
  netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
  netout[..., 5:] *= netout[..., 5:] > obj_thresh

  for i in range(grid_h*grid_w):
    row = i / grid_w
    col = i % grid_w
    for b in range(nb_box):
      # 4th element is objectness score
      objectness = netout[int(row)][int(col)][b][4]
      if(objectness <= obj_thresh).all(): continue
      # first 4 elements are x, y, w, and h
      x, y, w, h = netout[int(row)][int(col)][b][:4]
      x = (col + x) / grid_w # center position, unit: image width
      y = (row + y) / grid_h # center position, unit: image height
      w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
      h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
      # last elements are class probabilities
      classes = netout[int(row)][col][b][5:]
      box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
      boxes.append(box)
  return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
  new_w, new_h = net_w, net_h
  for i in range(len(boxes)):
    x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
    y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
    boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
    boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
    boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
    boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
  x1, x2 = interval_a
  x3, x4 = interval_b
  if x3 < x1:
    if x4 < x1:
      return 0
    else:
      return min(x2,x4) - x1
  else:
    if x2 < x3:
       return 0
    else:
      return min(x2,x4) - x3

def bbox_iou(box1, box2):
  intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
  intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
  intersect = intersect_w * intersect_h
  w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
  w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
  union = w1*h1 + w2*h2 - intersect
  return float(intersect) / union

def do_nms(boxes, nms_thresh):
  if len(boxes) > 0:
    nb_class = len(boxes[0].classes)
  else:
    return
  for c in range(nb_class):
    sorted_indices = np.argsort([-box.classes[c] for box in boxes])
    for i in range(len(sorted_indices)):
      index_i = sorted_indices[i]
      if boxes[index_i].classes[c] == 0: continue
      for j in range(i+1, len(sorted_indices)):
        index_j = sorted_indices[j]
        if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
          boxes[index_j].classes[c] = 0

# load and prepare an image
def load_image_pixels(filename, shape):
  # load the image to get its shape
  image = load_img(filename)
  width, height = image.size
  # load the image with the required size
  image = load_img(filename, target_size=shape)
  # convert to numpy array
  image = img_to_array(image)
  # scale pixel values to [0, 1]
  image = image.astype('float32')
  image /= 255.0
  # add a dimension so that we have one sample
  image = expand_dims(image, 0)
  return image, width, height

def load_image_cv(image, shape):
  width, height = image.shape[1], image.shape[0]
  image = cv2.resize(image, shape)
  image = image.astype('float32')
  image /= 255.0
  image = expand_dims(image, 0)
  return image, width, height

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
  v_boxes, v_labels, v_scores = list(), list(), list()
  # enumerate all boxes
  for box in boxes:
    # enumerate all possible labels
    for i in range(len(labels)):
      # check if the threshold for this label is high enough
      if box.classes[i] > thresh:
        v_boxes.append(box)
        v_labels.append(labels[i])
        v_scores.append(box.classes[i]*100)
        # don't break, many labels may trigger for one box
  return v_boxes, v_labels, v_scores

# draw all results
def draw_boxes(imdata, v_boxes, v_labels, v_scores, labels, all_colors):
  # load the image
  #data = pyplot.imread(filename)
  # plot the image
  fig, ax = pyplot.subplots()
  ax = pyplot.imshow(imdata)
  # get the context for drawing boxes
  ax = pyplot.gca()
  # plot each box
  for i in range(len(v_boxes)):
    box = v_boxes[i]
    # get coordinates
    y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
    # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    # create the shape
    label_index = labels.index(v_labels[i])
    # we have more colors than labels, so this is fine
    rect = Rectangle((x1, y1), width, height, fill=False, color=all_colors[label_index])
    # draw the box
    ax.add_patch(rect)
    # draw text and score in top left corner
    label = "%s (%.3f)" % (v_labels[i], v_scores[i])
    ax.text(x1, y1, label, color='white', bbox=dict(facecolor='blue', alpha=0.3))
  fig.canvas.draw()
  #pyplot.show()
  annotated = np.array(fig.canvas.renderer.buffer_rgba(), dtype=np.uint8)
  pyplot.close('all')
  cv2.imshow('camera', cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))


parser = argparse.ArgumentParser(description='train the model')
parser.add_argument('--images', type=str, default='.', help='folder with image files')
parser.add_argument('--defdriver', action='store_true', help='use default system video driver instead of DSHOW')
args = parser.parse_args()

serport = 'COM3'
servoOut = 0
servoMin = 4000
servoMax = 8000
# default position
servoX = round((servoMin + servoMax) / 2)

has_servo = True
try:
  servo = maestro.Controller(serport)
except:
  print('Could not connect to controller on port', serport)
  has_servo = False

if has_servo:
  servo.setTarget(servoOut, servoX)

pygame.mixer.init()
pygame.mixer.music.load('pew.wav')

# ugly hack, lol
all_colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

cam_w, cam_h = 640, 480

# load yolov3 model
model = load_model('model.h5')
# define the expected input shape for the model
input_w, input_h = 416, 416
# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
  "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
  "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
  "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
  "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
  "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
  "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
  "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
# define the probability threshold for detected objects
class_threshold = 0.6

if args.defdriver:
  # use default system driver
  cap = cv2.VideoCapture(0)
else:
  # use DSHOW to get rid of the letterboxed format on some cameras
  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# use the smallest available resolution
ret = cap.set(3,cam_w)
ret = cap.set(4,cam_h)

# super-sketchy code, will break if non-images are there
(_, _, imfiles) = next(os.walk(args.images))
#for imfile in imfiles:
while(True):
  #print(imfile)
  # define our new photo
  #photo_filename = os.path.normpath(args.images + '/' + imfile)
  # load and prepare image
  t1 = int(round(time.time() * 1000))
  #image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
  _, cvimage = cap.read()
  cvRGBimage = cvimage[...,::-1]
  image, image_w, image_h = load_image_cv(cvRGBimage, (input_w, input_h))
  #print(image.shape, cvimage.shape)
  t2 = int(round(time.time() * 1000))
  imgltime = t2 - t1
  # make prediction
  t1 = int(round(time.time() * 1000))
  yhat = model.predict(image)
  t2 = int(round(time.time() * 1000))
  predtime = t2 - t1
  # summarize the shape of the list of arrays
  #print([a.shape for a in yhat])
  t1 = int(round(time.time() * 1000))
  boxes = list()
  for i in range(len(yhat)):
    # decode the output of the network
    boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
  # suppress non-maximal boxes
  do_nms(boxes, 0.5)
  # correct the sizes of the bounding boxes for the shape of the image
  correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
  # get the details of the detected objects
  v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
  t2 = int(round(time.time() * 1000))
  boxtime = t2 - t1
  # summarize what we found
  if len(v_boxes) > 0:
    found_person = False
    for i in range(len(v_boxes)):
      vbc = v_boxes[i]
      xmed = round((vbc.xmin + vbc.xmax) / 2)
      if not found_person and v_labels[i] == 'person':
        servoX = round(servoMin + 0.75 * (servoMax - servoMin) * (cam_w - xmed) / cam_w)
        if has_servo:
          servo.setTarget(servoOut, servoX)
        pygame.mixer.music.play()
        # only track the first person in current frame
        found_person = True
      # rounding numpy floats is weird
      print(v_labels[i], int(round(v_scores[i])), '\t', servoX, '\timld:', imgltime, '\tpred:', predtime, '\tbox:', boxtime)
  else:
    print('nothing')
  # draw what we found
  draw_boxes(cvRGBimage, v_boxes, v_labels, v_scores, labels, all_colors)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
