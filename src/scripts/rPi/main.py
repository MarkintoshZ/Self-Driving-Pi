import numpy as np
from PIL import Image
import os
from os import system as sys
from keras.models import load_model
import keras
from replay import ExperienceReplay
import random
import time
import cv2
import drive


# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

steering_angle = 0
steering_angles = [-0.45, 0.45]

encoder = load_model('../models/80-2 Encoder')
exp_replay = ExperienceReplay()
new_model = False
batch_size = 64
skip_learn = 1

if new_model:
    epsilon = 0.5
    epsilon_decrease = 0.1

    model = keras.Sequential()
    model.add(keras.layers.Dense(2, activation='selu', input_dim=2))
    model.add(keras.layers.Dense(8, activation='selu'))
    model.add(keras.layers.Dense(2))
    model.compile('adam', loss='mse', metrics=['mse'])
else:
    epsilon = 0
    epsilon_decrease = 0

    print("Loading model")
    model = keras.models.load_model('../Models/new.h5')
    model.compile('adam', loss='mse', metrics=['mse'])


def get_new_state():

    # Grab a single frame of video
    ret, frame = video_capture.read()

    # print(frame.shape)
    # while size(frame[0][0]) == 0:
    #     print('No camera data  check connection')
    frame = frame[200:, 160: 480]
    image = cv2.resize(frame, (10, 8), Image.ANTIALIAS)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    data = np.asarray(image).ravel()
    data = data / 255
    data = encoder.predict(data.reshape(1, -1))
    input_t = data.reshape((-1))

    reward = 0
    game_over = False
    start = True

    return input_t, reward, game_over, start



if __name__ == '__main__':

    e = 0
    # connected to unity
    try:
        while True:
            e += 1

            print("running")

            start = False
            game_over = False
            loss = 0.
            counter = 0
            # get initial input

            while not start:
                input_t, reward, game_over, start = get_new_state()
                time.sleep(0.05)
                continue

            print('Starting New Session')

            while not game_over:

                counter += 1

                input_tm1 = input_t
                # get next action
                if np.random.rand() <= epsilon:
                    action = random.randint(0, 1)
                else:
                    q_value = model.predict(np.expand_dims(input_tm1, 0))
                    if q_value[0][0] == np.nan:
                        print("nan in model output")
                        break

                    action = np.argmax(q_value[0])
                    print(action)

                # apply action, get rewards and new state
                print(action)
                # send_msg(str(steering_angles[action]))
                if action:
                    drive.right()
                else:
                    drive.left()

                input_t, reward, game_over, start = get_new_state()

                # store experience
                exp_replay.remember([input_tm1, action, reward, input_t], game_over)

                # if counter % skip_learn == 0:
                #     print('training')
                #     inputs, targets = exp_replay.get_batch(model, batch_size)
                #     loss = np.array(model.train_on_batch(inputs, targets)).sum()


            print('training')
            for _ in range(25):
                inputs, targets = exp_replay.get_batch(model, batch_size)
                loss = np.array(model.train_on_batch(inputs, targets)).sum()

            print("Epoch {:03d} | Loss {:.4f} | Epsilon {:.3f}".format(e, loss, epsilon))

            epsilon -= epsilon_decrease
    except KeyboardInterrupt:
        name = input("\nenter file name")
        if name != 'no':
            model.save(name)
