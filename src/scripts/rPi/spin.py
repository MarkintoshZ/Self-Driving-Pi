from gpiozero import Motor
from time import sleep

left_motor = Motor(14, 4)
right_motor = Motor(20, 26)
speed = 0.4

if __name__ == '__main__':
    left_motor.forward(speed)
    right_motor.backward(speed)
    sleep(10)
