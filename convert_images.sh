#!/usr/bin/env bash

rm -rf images/small
mkdir images/small

# better algorithm than simple "convert -resize"
# keeps ringing artifacts smaller
find images/original/* | cut -d / -f 3 | parallel \
  convert images/original/{} -colorspace RGB +sigmoidal-contrast 6.5,50% \
    -filter Lanczos -distort resize 120x90 \
    -sigmoidal-contrast 6.5,50% -colorspace Gray images/small/{}.png
