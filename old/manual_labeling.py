import matplotlib.pyplot as plt
from PIL import Image
import os
from pprint import pprint
import json


def clicker(event):
  # this is trash code, sorry
  global imgdata, nameshort, learned, plt

  learned = False
  if event.button.name == 'LEFT':
    icontent = 'human'
    learned = True
  if event.button.name == 'RIGHT':
    icontent = 'empty'
    learned = True
  if learned:
    imgdata.update({nameshort : icontent})
    plt.close()


def typer(event):
  global goback, goforward, plt

  goback = False
  goforward = False
  if event.key == 'escape':
    goforward = True
    plt.close()
  if event.key == ' ':
    goback = True
    plt.close()
  if event.key == 'q':
    exit()


imgdir = 'images'
origdir = imgdir + '/original'
imdf = imgdir + '/imgdata.json'
try:
  jfile = open(imdf)
  imgdata = json.load(jfile)
  jfile.close()
except:
  imgdata = {}

imgs = os.listdir(origdir)

i = 0
learned = False
goback = False
goforward = False
while i < len(imgs):
  iname = imgs[i]
  nameshort = iname.split('.')[0]
  if nameshort in imgdata.keys() and not goback:
    i = i + 1
    continue
  goback = False
  image = Image.open(origdir + '/' + iname)
  plt.imshow(image)
  xlab = nameshort + '    ' + str(int(100*i/len(imgs))) + '%'
  if nameshort in imgdata.keys():
    xlab = xlab + '    ' + imgdata[nameshort]
  else:
    xlab = xlab + '    ' + '(learning)'
  plt.xlabel(xlab)
  plt.connect('button_press_event', clicker)
  plt.connect('key_press_event', typer)
  plt.show()
  if learned:
    jfile = open(imdf, 'w')
    json.dump(imgdata, jfile)
    jfile.close()
  if goforward:
    i = 0
  if goback:
    i = i - 1
    continue
  i = i + 1
