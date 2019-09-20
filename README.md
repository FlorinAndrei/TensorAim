# TensorAim
fly a drone, run TensorFlow image recognition, recognize people, TF points a video camera at them, records video

Not much to see here yet.

This needs to be a TensorFlow model, running on a Raspberry Pi, on a quadcopter drone. Feed TF images from a tiny surveillance camera. Do image recognition - the model is trained to recognize people in general.

TF needs to have 3 outputs:

- person has been recognized (threshold between 0 and 1)
- x coordinate within the surveillance image of the center of mass of the person (integer)
- y coordinate

Use x and y to aim a gimbal with the video camera. Then start filming as long as there's a person in the frame.

## Current status

Building synthetic training data is complete - each image contains a fuzzy circle (the object that needs to be recognized) and many noise dots.

Figuring out the best model architecture using the synthetic data.

Collecting / organizing real-world training data has started.

Testing synthetic data models agains live camera video.

TensorFlow on Raspberry Pi is broken. Not going to fly a drone for now - let's run it on a laptop. Waiting for TF 2.0, hopefully it gets better.
