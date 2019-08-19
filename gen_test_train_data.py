#!/usr/bin/env python3


import numpy as np
import cairo
import random


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


width, height = 640, 480
tgrad, tgthick, tgstep = 50, 2, 10
snrad = 10

raster = np.zeros(shape=(height, width), dtype=np.uint32)
workspace = cairo.ImageSurface.create_for_data(raster, cairo.FORMAT_ARGB32, width, height)
cc = cairo.Context(workspace)

# fill with white
cc.set_source_rgb(1, 1, 1)
cc.paint()

# paint the target
x, y = fuzzy_circle(cc, tgrad, tgthick, tgstep, width, height)

# make it snow
snow(cc, snrad, int(width * height / 10000), width, height)

#raster = workspace.get_data()
#np.savetxt('raster.txt', raster)

workspace.write_to_png('test.png')

# TODO: pickle
