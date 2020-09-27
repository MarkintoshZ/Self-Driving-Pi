## Self Driving Pi

This is the repo for __John Papetti III__  and __Mark Zhang__'s self
driving car project for the Wardlaw + Hartridge School 2019
Family Science Night (FSN)

[![An Intro to Self Driving Car](https://img.youtube.com/vi/b_zeB34lh58/0.jpg)](https://www.youtube.com/watch?v=b_zeB34lh58)


In this project, a self driving robot is created using a Raspberry Pi 3. The
robot can successfully follow a road (black strip on the white surface) using
machine learning models. The program process the pixel data from a webcam and
control the robot's movement. A 80-to-2 dimension autoencoder was used to reduce
the dimension space of the environment. DQN, a reinforcement learning model, was
used to learn how to drive. The DQN network was trained in a computer simulation
before being implemented on the actual robot.
