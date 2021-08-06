
import gym
from gym import logger as gymlogger
gymlogger.set_level(40) #error only
from pyglet.window import key
import numpy as np
import argparse
import pathlib


from customPong import CustomPong
from neural_network import SlowNeuralNet, FastNeuralNet
from utils import displaySingleAgentGame


def playNN():  
    a = np.array([0.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 0.0])

    def key_press(k, mod):
        if k == key.Q:
            a[0] += 1.0
            a[2] -= 1.0
        if k == key.W:
            a[0] -= +1.0
            a[2] += +1.0
        if k == key.O:
            a[1] += +1.0
            a[3] -= +1.0
        if k == key.P:
            a[1] -= +1.0
            a[3] += +1.0

    def key_release(k, mod):
        if k == key.Q:
            a[0] = 0
            a[2] = 0
        if k == key.W:
            a[0] = 0
            a[2] = 0
        if k == key.O:
            a[1] = 0
            a[3] = 0
        if k == key.P:
            a[1] = 0
            a[3] = 0
    GAME = 'BipedalWalker-v3'
    env = gym.make(GAME)
    ob = env.reset()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    totalReward = 0
    for step in range(100000000):
        env.render()

        ob, reward, done, info = env.step(a)
        totalReward += reward
        if done:
            break
    ob = env.reset()
    print("Expected Fitness of %4d | Actual Fitness = %2f" % (0, totalReward))
    env.close()

def get_generation(str):
    return float(str.split('gen_')[1].split('.')[0])

if __name__ == "__main__":
    playNN()