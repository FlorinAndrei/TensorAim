# TensorAim

TensorFlow sentry, recognize people, point a laser at them, make pew-pew-pew sound.

TF needs to have 3 outputs:

- person has been recognized (threshold between 0 and 1)
- xmin coordinate within the surveillance image of the person
- xmax coordinate

Use xmin and xmax to aim the laser within the horizontal plane. Then trigger pew-pew-pew sound.

ymin-ymax could be used to aim vertically, but building the full gimbal takes more time. Perhaps that's v2.0.

## Current status

TensorAim can now detect humans in a live video stream. It can also determine xmin/xmax. Ready to connect to servo for aiming.

## Previous status updates (newest to oldest)

TensorFlow on Raspberry Pi is broken. Not going to fly a drone for now - let's run it on a laptop. Waiting for TF 2.0, hopefully it gets better.

Testing synthetic data models agains live camera video.

Collecting / organizing real-world training data has started.

Figuring out the best model architecture using the synthetic data.

Building synthetic training data is complete - each image contains a fuzzy circle (the object that needs to be recognized) and many noise dots.
