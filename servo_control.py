import maestro

servo = maestro.Controller('COM3')

servo.setAccel(0,4)      #set servo 0 acceleration to 4
servo.setTarget(0,6000)  #set servo to move to center position
servo.setSpeed(1,10)     #set speed of servo 1
#x = servo.getPosition(1) #get the current position of servo 1

for i in range(0, 23):
  print(i, servo.getMin(i), servo.getMax(i))

# well, at least this works
print(dir(servo.Targets))

servo.close()
