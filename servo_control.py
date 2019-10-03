import maestro
import time

serport = 'COM3'

try:
  servo = maestro.Controller(serport)
except:
  print('Could not connect to controller on port', serport)
  exit(1)

while True:
  servo.setTarget(0, 4000)
  time.sleep(1)
  servo.setTarget(0, 5000)
  time.sleep(1)
  servo.setTarget(0, 6000)
  time.sleep(1)
  servo.setTarget(0, 7000)
  time.sleep(1)
  servo.setTarget(0, 8000)
  time.sleep(1)

if servo.isMoving(0):
  print('moving')
else:
  print('stopped')

print(servo.getPosition(0))

servo.close()
