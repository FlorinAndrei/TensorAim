# TensorAim

The system needs to do two things:

- object recognition and classification
- point out the coordinates in space of the recognized objects

With that, action can be taken in the real world. E.g., pick one object category and literally point it out - paint a dot on them with a laser.

## Technology

The `sentry.py` file instantiates a deep neural network, feeds the network live images from a camera, parses the output, and estimates the locations of the detected objects (if any).

The model can provide 2D coordinates (X/Y) for the objects; `sentry.py` uses that to draw bounding boxes around the objects. Currently only the X coordinate (horizontal plane) is passed beyond the software realm into the hardware; 2D control (X/Y) would be doable, but that's for a future version.

The software can control a servo mechanism in real time, via standard PWM protocols, to point a laser at the objects that are detected and localized. Currently the X coordinate from object detection is used to swivel the laser left-right.

`train.py` parses the YOLOv3 weights and compiles them into a format compatible with TensorFlow / Keras.

## Hardware details

We rely on simple technology used for amateur R/C (radiocontrolled) model vehicles (cars, planes, helicopters, drones). The AI software runs on a regular computer, and controls a servo which points a laser in the direction of the detected object.

The laser sits on top of a [Hextronik HXT900](https://servodatabase.com/servo/hextronik/hxt900) servo. It's a cheap, small servo typically used for R/C planes.

The interface between servo and computer is provided by the [Pololu Mini Maestro 12-Channel](https://www.pololu.com/product/1352) USB servo controller. The controller has 12 outputs and can control up to 12 independent servos. Each output speaks the PWM protocol typically used by R/C gear. The controller input is USB/serial and is plugged into the computer.

Power for the servo is provided, as is standard with R/C, by a LiPo battery via an ESC BEC.

Here's an image of the hardware:

![hardware](https://raw.githubusercontent.com/FlorinAndrei/TensorAim/master/docs/hw_photo.jpg)

## Current status

Target detection, aiming with servo - ready for testing with laser.

## Credits

The system is based on the YOLOv3 model by [Joseph Redmon](https://pjreddie.com/). Unlike R-CNN, it uses a single network to look at the whole image. It's extremely fast, while remaining accurate enough. Real time object tracking at video frame rates is doable with YOLO on consumer hardware.

YOLO output parsing code was borrowed from [Huynh Ngoc Anh a.k.a. experiencor](https://github.com/experiencor).

Hardware for the laser mount was designed and built by Victor Andrei.

## Previous status updates (newest to oldest)

TensorAim can now detect humans in a live video stream. It can also determine xmin/xmax. Ready to connect to servo for aiming.

TensorFlow on Raspberry Pi is broken. Not going to fly a drone for now - let's run it on a laptop. Waiting for TF 2.0, hopefully it gets better.

Testing synthetic data models agains live camera video.

Collecting / organizing real-world training data has started.

Figuring out the best model architecture using the synthetic data.

Building synthetic training data is complete - each image contains a fuzzy circle (the object that needs to be recognized) and many noise dots.
