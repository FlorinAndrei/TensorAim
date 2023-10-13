from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image, ImageDraw, ImageColor
import argparse, textwrap
import cv2
import time
# https://github.com/FRC4564/Maestro
import maestro
import pygame

# fix issue: "Could not create cudnn handle"
# may not be needed on all systems
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
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
  intersect_w = _interval_overlap([box1.xmin, box1.xmax],
    [box2.xmin, box2.xmax])
  intersect_h = _interval_overlap([box1.ymin, box1.ymax],
    [box2.ymin, box2.ymax])
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


def load_image_cv(image, shape):
  width, height = image.shape[1], image.shape[0]
  image = cv2.resize(image, shape)
  # we got int, we need float
  image = image.astype('float32')
  # normalize
  image /= 255.0
  # YOLO expects more dimensions
  image = np.expand_dims(image, 0)
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


def draw_boxes(imdata, v_boxes, v_labels, v_scores, labels):
  pilim = Image.fromarray(imdata)
  draw = ImageDraw.Draw(pilim)
  for i in range(len(v_boxes)):
    box = v_boxes[i]
    # get coordinates
    y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
    # calculate width and height of the box
    width, height = x2 - x1, y2 - y1
    # create the shape and apply
    label_index = labels.index(v_labels[i])
    label = "%s (%d)" % (v_labels[i], round(v_scores[i]))
    objcolor = list(ImageColor.colormap.keys())[label_index]
    draw.rectangle((x1, y1, x1 + width, y1 + height), fill=None,
      outline=objcolor, width=1)
    draw.text((x1 + 3, y1 + 1), label, fill=objcolor)
  # RGB, BGR - different orders are used by PIL and OpenCV
  im4cv2 = cv2.cvtColor(np.array(pilim), cv2.COLOR_BGR2RGB)
  # zoom the image for convenience
  im4cv2big = cv2.resize(im4cv2, None, fx = 1.7, fy = 1.7)
  # display annotated image
  cv2.imshow('camera', im4cv2big)


servoMin = 4000
servoMax = 8000
servoMed = round((servoMin + servoMax) / 2)

parser = argparse.ArgumentParser(
  description=textwrap.dedent("""\
  Run the model, detect objects, estimate positions, control servo.

  Hotkeys:
  -     q: quit
  - space: reset servo to center
  """),
  formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--center', type=int, default=servoMed,
  help='servo initial center, to correct servo/camera offset, between ' +
    str(servoMin) + ' and ' + str(servoMax) + ', default=' + str(servoMed))
parser.add_argument('--amplix', type=float, default=0.7,
  help='motion amplitude on X; depends on camera angle and servo; default=0.7')
parser.add_argument('--defdriver', action='store_true',
  help='use default system video driver instead of DSHOW')
parser.add_argument('--serport', type=str, default='COM3',
  help='serial port to use for servo; default: COM3')
args = parser.parse_args()

# which output to use on servo
servoOut = 0
# default position
servoX = args.center
has_servo = True
try:
  servo = maestro.Controller(args.serport)
except:
  print('Could not connect to controller on port', args.serport)
  has_servo = False
# reset servo to initial position
if has_servo:
  servo.setTarget(servoOut, servoX)

pygame.mixer.init()
pygame.mixer.music.load('pew.wav')

# load yolov3 model
model = load_model('model.h5')
print(model.summary())
# this is the image size that the model expects
input_w, input_h = 416, 416
# define the labels
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
  "truck", "boat", "traffic light", "fire hydrant", "stop sign",
  "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
  "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
  "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
  "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
  "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
  "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
  "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
# define the anchors
anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119],
  [10,13, 16,30, 33,23]]
# define the probability threshold for detected objects
class_threshold = 0.6

# no need to capture in hi-def if the model can't do that
cam_w, cam_h = 640, 480
if args.defdriver:
  # use default system driver
  cap = cv2.VideoCapture(0)
else:
  # use DSHOW to get rid of the letterboxed format on some cameras
  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# use the smallest available resolution
ret = cap.set(3,cam_w)
ret = cap.set(4,cam_h)

while(True):
  # load and prepare image
  _, cvimage = cap.read()
  cvRGBimage = cvimage[...,::-1]
  image, image_w, image_h = load_image_cv(cvRGBimage, (input_w, input_h))
  # make prediction
  t1 = int(round(time.time() * 1000))
  yhat = model.predict(image)
  t2 = int(round(time.time() * 1000))
  predtime = t2 - t1
  t1 = int(round(time.time() * 1000))
  boxes = list()
  for i in range(len(yhat)):
    # decode the output of the network
    boxes += decode_netout(yhat[i][0], anchors[i], class_threshold,
      input_h, input_w)
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
        # translate image coordinates into servo coordinates
        servoX = round(args.center - args.amplix * (servoMax - servoMed) *
          (xmed - cam_w / 2) / (cam_w / 2))
        # just in case servoX is out of bounds
        if servoX < servoMin:
          servoX = servoMin
        if servoX > servoMax:
          servoX = servoMax
        if has_servo:
          servo.setTarget(servoOut, servoX)
        pygame.mixer.music.play()
        # only track the first person in current frame
        found_person = True
      # rounding numpy floats is weird
      print(v_labels[i], int(round(v_scores[i])), '\t', servoX,
        '\tpred:', predtime, 'ms\tbox:', boxtime, 'ms')
  else:
    print('nothing')
  # draw what we found
  draw_boxes(cvRGBimage, v_boxes, v_labels, v_scores, labels)
  # read keyboard
  k = cv2.waitKey(1)
  if k == ord('q'):
    break
  if k == ord(' '):
    servoX = args.center
