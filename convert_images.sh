#!/usr/bin/env bash

rm -rf images/small
mkdir images/small

# better algorithm than simple "convert -resize"
# keeps ringing artifacts smaller
# https://imagemagick.org/discourse-server/viewtopic.php?f=1&t=36709
find images/original/* | cut -d / -f 3 | parallel \
  convert images/original/{} -colorspace Gray \
    -filter Triangle -resize 120x90 \
    images/small/{}.png
