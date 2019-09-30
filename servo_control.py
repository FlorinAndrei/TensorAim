import maestro

servo = maestro.Controller('COM3')

servo.setTarget(0,1500)  #set servo to move to center position
servo.setTarget(0,1200)
servo.setTarget(0,1800)
servo.setTarget(0,1500)

servo.close()
