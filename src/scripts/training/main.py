import numpy as np
from multiprocessing import Queue
import server
from server import send_msg
from PIL import Image
import os
from os import system as sys
from keras.models import load_model
import keras
from replay import ExperienceReplay
import random
import time

q = Queue()
server.start(q)

# the simulator saves the screen shots in to this folder
# for the python script to read
screen_shot_path = '../../ScreenShots'
sys('rm -rf ' + screen_shot_path + '; mkdir ' + screen_shot_path)

steering_angle = 0
steering_angles = [-0.45, 0.45]

encoder = load_model('../../models/80-2 encoder')
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
    model = keras.models.load_model('../../models/dqn_0.h5')
    model.compile('adam', loss='mse', metrics=['mse'])


def get_new_state():
    while True:
        try:
            screen_shots = os.listdir(screen_shot_path)
            try:
                screen_shots.remove('.DS_Store')
            except ValueError:
                pass

            try:
                image_path = str(max([float(fn[:-4]) for fn in screen_shots])) + '.png'
            except ValueError:
                continue

            image = Image.open(os.path.join(screen_shot_path, image_path)).convert('L')
            # print('rm ../ScreenShots/' + image_screen_shot_path)
            print(sys('rm -rf ' + screen_shot_path + '/*'))
            # image.show()
            break
        except OSError:
            time.sleep(0.05)

    image = image.resize((10, 8), Image.ANTIALIAS)
    data = np.asarray(image).ravel()
    data = data / 255
    data = encoder.predict(data.reshape(1, -1))
    input_t = data.reshape((-1))

    reward = 0
    game_over = False
    start = False
    while not q.empty():
        queue = q.get()

        if 'reward' in queue:
            reward = info.split('=')[1]

        if queue == 'gameover':
            game_over = True
            reward = -1

        if queue == 'start':
            start = True

        time.sleep(0.05)

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

                # apply action, get rewards and new state
                print(action)
                send_msg(str(steering_angles[action]))

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
        name = input("enter file name")
        if name != 'no':
            model.save(name)
