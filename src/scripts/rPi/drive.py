from gpiozero import Motor
from time import sleep


left_motor = Motor(14, 4)
right_motor = Motor(20, 26)
SPEED = 0.65
DRIVE_TIME = 0.05
WAIT_TIME = 0.1

last_state = -1


def left():
    global last_state

    if last_state == 1 or last_state == -1:
        left_motor.forward(speed=SPEED)
        sleep(DRIVE_TIME)
        right_motor.stop()
        sleep(WAIT_TIME)

    last_state = 0


def right():
    global last_state

    if last_state == 0 or last_state == -1:
        right_motor.forward(speed=SPEED)
        sleep(DRIVE_TIME)
        left_motor.stop()
        sleep(WAIT_TIME)

    last_state = 1


if __name__ == '__main__':

    print('works')
