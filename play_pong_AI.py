
from gym import logger as gymlogger
gymlogger.set_level(40) #error only
from pyglet.window import key
import numpy as np


from customPong import CustomPong
from neural_network import SlowNeuralNet, FastNeuralNet


def playNN(nn):  
    a = np.array([0.0, 0.1, 0.0])

    def key_press(k, mod):
        if k == key.UP:
            a[0] = +1.0
        if k == key.DOWN:
            a[2] = +1.0

    def key_release(k, mod):
        if k == key.UP:
            a[0] = 0
        if k == key.DOWN:
            a[2] = 0

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
            al = a
            ar = nn.getOutput(ob_r)
        (ob_l, ob_r), (r_l, r_r), done, info = env.step(al, ar)
        # print(f"{ob_l} - {ob_r}")
        totalReward += r_r
        if done:
            break
    observation = env.reset()
    print("Expected Fitness of %4d | Actual Fitness = %2f" % (nn.fitness, totalReward))
    env.close()



nn = SlowNeuralNet([1])
nn.from_json("saved_models/bestnn.json")
# with open("bestnn_g.json", "rb") as f:
# with open("saved_models/bestnn.json", "rb") as f:
#     nn_data = pickle.load(f)
playNN(nn)
# show_video()