import numpy as np
import json
from PIL import Image
import h5py

import config


imgdir = 'images'
smalldir = imgdir + '/small'
imdf = imgdir + '/imgdata.json'

jfile = open(imdf, 'r')
imgdata = json.load(jfile)
jfile.close()

images = np.empty((len(imgdata), config.imgh, config.imgw, 1), dtype=np.float32)
labels = np.empty((len(imgdata)), dtype=np.uint8)

i = 0
for k,v in imgdata.items():
  if v == 'human':
    label = 1
  elif v == 'empty':
    label = 0
  else:
    print('bad label')
    print('i:', i, '    k:', k, '    v:', v)
    exit(1)
  labels[i] = label
  imgfile = smalldir + '/' + k + '.JPG.png'
  
  imgraster = Image.open(imgfile)
  imgraster.load()
  imgnp = np.asarray(imgraster, dtype=np.float32)
  imgnp = np.reshape(imgnp, (config.imgh, config.imgw, 1))
  images[i, :, :, :] = imgnp
  imgraster.close()

  i += 1

hf = h5py.File('real_world_data.h5', 'w')
hf.create_dataset('images', data=images)
hf.create_dataset('labels', data=labels)
hf.close()
