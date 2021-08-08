
from gym import logger as gymlogger
gymlogger.set_level(40) #error only
from pyglet.window import key
import numpy as np


from customPong import CustomPong


def playHuman():  
    action_r = np.array([0.0, 0.1, 0.0])
    action_l = np.array([0.0, 0.1, 0.0])

    def key_press(k, mod):
        if k == key.UP:
            action_r[0] = +1.0
        if k == key.W:
            action_l[0] = +1.0
        if k == key.DOWN:
            action_r[2] = +1.0
        if k == key.S:
            action_l[2] = +1.0

    def key_release(k, mod):
        if k == key.UP:
            action_r[0] = 0
        if k == key.W:
            action_l[0] = 0
        if k == key.DOWN:
            action_r[2] = 0
        if k == key.S:
            action_l[2] = 0

    env = CustomPong()
    ob_l, ob_r = env.reset()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    totalReward = 0
    for step in range(100000000):
        # if step%3 == 0:
        env.render()
        if step % 10 == 0:
            al = action_l
            ar = action_r
        (ob_l, ob_r), (r_l, r_r), done, info = env.step((al, ar))
        # print(f"{ob_l} - {ob_r}")
        totalReward += r_r
        if done:
            break
    observation = env.reset()
    env.close()

playHuman()
# show_video()