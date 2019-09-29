import numpy as np
import cairo
import random
from PIL import Image
import pickle
import argparse
import math
import gc
from pprint import pprint

import config


def snow(ctext, maxrad, num_flakes, w, h):
  for i in range(num_flakes):
    x, y = find_center(maxrad, maxrad, w, h)
    for j in range(maxrad, 0, -1):
      col = j / (maxrad + 1)
      ctext.arc(x, y, j, 0, 2 * np.pi)
      ctext.set_source_rgb(col, col, col)
      ctext.fill()
      ctext.stroke()


def sharp_circle(ctext, rad, thick, col, x, y, fill):
  ctext.arc(x, y, rad, 0, 2 * np.pi)
  ctext.set_line_width(thick)
  ctext.set_source_rgb(col, col, col)
  if fill:
    ctext.fill()
  ctext.stroke()


def sharp_square(ctext, halfw, thick, col, x, y):
  ctext.rectangle(x-halfw, y-halfw, 2*halfw, 2*halfw)
  ctext.set_line_width(thick)
  ctext.set_source_rgb(col, col, col)
  ctext.stroke()


def fuzzy_square(ctext, mhalfw, thick, steps, w, h):
  x, y = find_center(mhalfw, steps, w, h)
  for i in range(steps):
    col = i / (steps - 1)
    sharp_square(ctext, mhalfw-i, thick, col, x, y)
    sharp_square(ctext, mhalfw+i, thick, col, x, y)
  return x, y


def find_center(mrad, steps, w, h):
  wmin = steps + mrad
  wmax = w - steps - mrad
  hmin = steps + mrad
  hmax = h - steps - mrad
  x = random.randrange(wmin, wmax)
  y = random.randrange(hmin, hmax)
  return x, y


def fuzzy_circle(ctext, mrad, thick, steps, w, h):
  x, y = find_center(mrad, steps, w, h)
  for i in range(steps):
    col = i / (steps - 1)
    sharp_circle(ctext, mrad-i, thick, col, x, y, False)
    sharp_circle(ctext, mrad+i, thick, col, x, y, False)
  return x, y


parser = argparse.ArgumentParser(description='generate synthetic training data')
parser.add_argument('--snow', type=int, default=100, help='snow factor, default=100, bigger means less snow')
args = parser.parse_args()

# target line thickness
tgthick = 1
# target steps
tgstepmin, tgstepmax = 2, 10
# target radius
tgradmin, tgradmax = 8, math.floor(config.imgh/2 - tgstepmax - 1)
# snow radius
snrad = 4
setsize = 10000

raster = np.zeros(shape=(config.imgh, config.imgw), dtype=np.uint32)
workspace = cairo.ImageSurface.create_for_data(raster, cairo.FORMAT_RGB24, config.imgw, config.imgh)
cc = cairo.Context(workspace)

print('Generating image set with', setsize, 'images:')

# keep everything in a list
train_set = []

for step in range(setsize):
  print(step, end='\r', flush=True)
  # fill with white
  cc.set_source_rgb(1, 1, 1)
  cc.paint()
  
  obj_kind = np.random.randint(2)
  tgrad = random.randint(tgradmin, tgradmax)
  tgstep = random.randint(tgstepmin, tgstepmax)
  if tgstep >= tgrad:
    tgstep = tgrad - 1
  if obj_kind == 0:
    # square
    x, y = fuzzy_square(cc, tgrad, tgthick, tgstep, config.imgw, config.imgh)
    obj_kind = 0
  elif obj_kind == 1:
    # round target
    x, y = fuzzy_circle(cc, tgrad, tgthick, tgstep, config.imgw, config.imgh)
    obj_kind = 1
  else:
    print('wrong value for obj_kind:', obj_kind)
    exit()
  
  snow(cc, snrad, int(config.imgw * config.imgh / args.snow), config.imgw, config.imgh)
  
  # convert the workspace to a monochrome 8 bit image
  
  # make an output raster with all pixels decomposed as RGB triplets
  buf = workspace.get_data()
  rasout = np.ndarray(shape=(config.imgh, config.imgw, 4), dtype=np.uint8, buffer=buf)
  # slice off the unused alpha channel
  rasout = rasout[:, :, :3]
  # convert to mono by flattening the 3D matrix
  rasout = np.mean(rasout, axis=2, keepdims=True)
  # maximize contrast by expanding the range to 0-255
  rmin = np.min(rasout)
  rmax = np.max(rasout)
  output = np.empty(np.shape(rasout), dtype=np.uint8)
  output = np.interp(rasout, [rmin, rmax], [0, 255]).astype(np.uint8)
  
  # save image
  #im = Image.fromarray(output, 'L')
  #im.save('test.png')
  
  # Make one training object - a list with:
  # - whether the target exists or not
  # - the coordinates of the center of the target
  # - the whole image
  train_obj = []
  obj_kind_arr = np.zeros((2), dtype=np.uint8)
  obj_kind_arr[obj_kind] = 1
  train_obj.append(obj_kind_arr)
  # coordinates require a model that can train for that
  train_obj.append([x / config.imgw, y / config.imgh])
  train_obj.append(output)
  # append the object to the training set
  # bug with garbage collection makes append() very slow; turn it off
  gc.disable()
  train_set.append(train_obj)
  gc.enable()

# pickle the set
pickle.dump(train_set, open('synthetic_data.p', 'wb'))

print('All done.')
