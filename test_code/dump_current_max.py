import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
from numpy.core.fromnumeric import argmax
gymlogger.set_level(40) #error only
# import tensorflow as tf
import numpy as np
import random
import pickle
from pyglet.window import key
import json

def wrap_env(env):
  env = Monitor(env, './video', force=True, video_callable=lambda episode_id: True)
  return env

import time, math, random, bisect, copy
import gym
import numpy as np


from customPong import CustomPong



class NeuralNet : 
    def __init__(self, nodeCount, seed):     
        self.fitness = 0.0
        self.nodeCount = nodeCount
        self.weights = []
        self.biases = []
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        for i in range(len(nodeCount) - 1):
            self.weights.append( self.rng.uniform(low=-1, high=1, size=(nodeCount[i], nodeCount[i+1])).tolist() )
            self.biases.append( self.rng.uniform(low=-1, high=1, size=(nodeCount[i+1])).tolist())


    def printWeightsandBiases(self):
        
        print("--------------------------------")
        print("Weights :\n[", end="")
        for i in range(len(self.weights)):
            print("\n [ ", end="")
            for j in range(len(self.weights[i])):
                if j!=0:
                    print("\n   ", end="")
                print("[", end="")
                for k in range(len(self.weights[i][j])):
                    print(" %5.2f," % (self.weights[i][j][k]), end="")
                print("\b],", end="")
            print("\b ],")
        print("\n]")

        print("\nBiases :\n[", end="")
        for i in range(len(self.biases)):
            print("\n [ ", end="")
            for j in range(len(self.biases[i])):
                    print(" %5.2f," % (self.biases[i][j]), end="")
            print("\b],", end="")
        print("\b \n]\n--------------------------------\n")
  
    def getOutput(self, input):
        output = input
        for i in range(len(self.nodeCount)-1):
            # output = np.reshape( np.maximum(np.matmul(output, self.weights[i]) + self.biases[i], 0), (self.nodeCount[i+1]))
            output = np.reshape(np.matmul(output, self.weights[i]) + self.biases[i], (self.nodeCount[i+1]))
        return output
    
    def to_json(self, fn):
        nn_data = {
        "fitness": self.fitness,
            "nodeCount": self.nodeCount,
            "weights": self.weights,
            "biases": self.biases,
            "seed": self.seed,
        }
        with open(fn, 'w+') as f:
            json.dump(nn_data, f)
    
    def from_json(self, fn):
        with open(fn, 'w+') as f:
            nn_data = json.load(fn)
        
        self.fitness = nn["fitness"]
        self.nodeCount = nn["nodeCount"]
        self.weights = nn["weights"]
        self.biases = nn["biases"]
        self.seed = nn["seed"]
        self.rng = np.random.RandomState(self.seed)

class Population :
    def __init__(self, populationCount, mutationRate, nodeCount):
        self.nodeCount = nodeCount
        self.popCount = populationCount
        self.m_rate = mutationRate
        self.population = [ NeuralNet(nodeCount, np.random.randint(10000)) for i in range(populationCount)]


    def createChild(self, nn1, nn2):
        
        child = NeuralNet(self.nodeCount, np.random.randint(10000))
        total_fitness = nn1.fitness + nn2.fitness
        for i in range(len(child.weights)):
            for j in range(len(child.weights[i])):
                for k in range(len(child.weights[i][j])):
                    if random.random() > self.m_rate:

                        if random.random() < (nn1.fitness / (nn1.fitness+nn2.fitness)) if total_fitness > 0 else 0:
                            child.weights[i][j][k] = nn1.weights[i][j][k]
                        else :
                            child.weights[i][j][k] = nn2.weights[i][j][k]
                        

        for i in range(len(child.biases)):
            for j in range(len(child.biases[i])):
                if random.random() > self.m_rate:
                    if random.random() < nn1.fitness / (nn1.fitness / (nn1.fitness+nn2.fitness)) if total_fitness > 0 else 0:
                        child.biases[i][j] = nn1.biases[i][j]
                    else:
                        child.biases[i][j] = nn2.biases[i][j]

        return child


    def createNewGeneration(self, bestNN):    
        nextGen = []
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        for i in range(self.popCount):
            if random.random() < float(self.popCount-i)/self.popCount:
                nextGen.append(copy.deepcopy(self.population[i]));

        fitnessSum = [0]
        minFit = min([i.fitness for i in nextGen])
        for i in range(len(nextGen)):
            fitnessSum.append(fitnessSum[i]+(nextGen[i].fitness-minFit)**4)
        

        while(len(nextGen) < self.popCount):
            r1 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            r2 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            i1 = bisect.bisect_left(fitnessSum, r1)
            i2 = bisect.bisect_left(fitnessSum, r2)
            if 0 <= i1 < len(nextGen) and 0 <= i2 < len(nextGen) :
                nextGen.append( self.createChild(nextGen[i1], nextGen[i2]) )
            else :
                print("Index Error ");
                print("Sum Array =",fitnessSum)
                print("Randoms = ", r1, r2)
                print("Indices = ", i1, i2)
        self.population.clear()
        self.population = nextGen


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
        if step%3 == 0:
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



nn = None
# with open("bestnn_g.json", "rb") as f:
with open("bestnn.json", "rb") as f:
    nn = pickle.load(f)
nn_data = {
    "fitness": nn.fitness,
        "nodeCount": nn.nodeCount,
        "weights": nn.weights,
        "biases": nn.biases,
        "seed": nn.seed if hasattr(nn, "seed") else 0,
}

with open("bestnn_j.json", "w+") as f:
    json.dump(nn_data, f)

# show_video()