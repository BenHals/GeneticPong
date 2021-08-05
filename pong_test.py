import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
gymlogger.set_level(40) #error only
# import tensorflow as tf
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import math
import glob
import io
import base64
from IPython.display import HTML

from IPython import display as ipythondisplay

import cv2

from customPong import CustomPong


env = CustomPong()
print(env.action_space)
for i in range(1000):
    env.render()
    observation, reward, done, d = env.step(env.action_space.sample())
    print(reward)
    if done:
        break

# input()