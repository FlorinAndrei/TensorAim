#!/usr/bin/env python3


import numpy as np
import cairo
import random
from PIL import Image
from pprint import pprint


def snow(ctext, maxrad, num_flakes, w, h):
  for i in range(num_flakes):
    x, y = find_center(maxrad, maxrad, w, h)
    for j in range(maxrad, 0, -1):
      col = j / (maxrad - 1)
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


width, height = 120, 90
tgrad, tgthick, tgstep = 10, 1, 3
snrad = 4

raster = np.zeros(shape=(height, width), dtype=np.uint32)
workspace = cairo.ImageSurface.create_for_data(raster, cairo.FORMAT_RGB24, width, height)
cc = cairo.Context(workspace)

# fill with white
cc.set_source_rgb(1, 1, 1)
cc.paint()

# paint the target
x, y = fuzzy_circle(cc, tgrad, tgthick, tgstep, width, height)

# make it snow
snow(cc, snrad, int(width * height / 1000), width, height)


# convert the workspace to a monochrome 8 bit image

# make an output raster with all pixels decomposed as RGB triplets
buf = workspace.get_data()
rasout = np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=buf)
# slice off the unused alpha channel
rasout = rasout[:, :, :3]
# convert to mono by flattening the 3D matrix
rasout = np.mean(rasout, axis=2)
# maximize contrast by expanding the range to 0-255
rmin = np.min(rasout)
rmax = np.max(rasout)
output = np.empty(np.shape(rasout), dtype=np.uint8)
output = np.interp(rasout, [rmin, rmax], [0, 255]).astype(np.uint8)

# save image
im = Image.fromarray(output, 'L')
im.save('test.png')

# TODO: pickle
