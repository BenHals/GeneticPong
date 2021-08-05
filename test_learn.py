import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
from numpy.core.fromnumeric import argmax
gymlogger.set_level(40) #error only
# import tensorflow as tf
import numpy as np
import random
import pickle
import json


def wrap_env(env):
  env = Monitor(env, './video', force=True, video_callable=lambda episode_id: True)
  return env

import time, math, random, bisect, copy
import gym
import numpy as np
import tqdm


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
        with open(fn, 'r') as f:
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


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


# def replayBestBots(bestNeuralNets, steps, sleep):  
#     choice = input("Do you want to watch the replay ?[Y/N] : ")
#     if choice=='Y' or choice=='y':
#         for i in range(len(bestNeuralNets)):
#             if (i+1)%steps == 0 :
#                 observation = env.reset()
#                 totalReward = 0
#                 for step in range(MAX_STEPS):
#                     env.render(totalReward)
#                     time.sleep(sleep)
#                     action = bestNeuralNets[i].getOutput(observation)
#                     observation, reward, done, info = env.step(action)
#                     totalReward += reward
#                     if done:
#                         observation = env.reset()
#                         break
#                 print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %2f" % (i+1, bestNeuralNets[i].fitness, totalReward))


def recordBestBots(bestNeuralNets):  
    print("\n Recording Best Bots ")
    print("---------------------")
    # env = wrap_env(gym.make('BipedalWalker-v3'))
    env = CustomPong()
    ob_l, observation = env.reset()
    # for i in range(len(bestNeuralNets)):
    i = len(bestNeuralNets) - 1
    totalReward = 0
    for step in range(MAX_STEPS):
        env.render()
        if step % 10 == 0:
            action = bestNeuralNets[i].getOutput(observation)
            al = bestNeuralNets[i].getOutput(ob_l)
        (ob_l, observation), (r_l, reward), done, info = env.step(al, action)
        totalReward += reward
        if done:
            break
    observation = env.reset()
    print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %2f" % (i+1, bestNeuralNets[i].fitness, totalReward))
    bestNeuralNets[i].printWeightsandBiases()
    env.close()

def mapRange(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.

    return rightMin + (valueScaled * rightSpan)

def normalizeArray(aVal, aMin, aMax): 
    res = []
    for i in range(len(aVal)):
        res.append( mapRange(aVal[i], aMin[i], aMax[i], -1, 1) )
    return res

def scaleArray(aVal, aMin, aMax):   
    res = []
    for i in range(len(aVal)):
        res.append( mapRange(aVal[i], -1, 1, aMin[i], aMax[i]) )
    return res


GAME = 'BipedalWalker-v3'
MAX_STEPS = 15000
MAX_GENERATIONS = 1000
POPULATION_COUNT = 500
MUTATION_RATE = 0.02
COMPARISON_TESTS = 50
# env = gym.make(GAME)
env = CustomPong()
observation = env.reset()
in_dimen = env.observation_space.shape[0]
try:
    out_dimen = env.action_space.shape[0]
except:
    out_dimen = env.action_space.n
# obsMin = env.observation_space.low
# obsMax = env.observation_space.high
# actionMin = env.action_space.low
# actionMax = env.action_space.high
pop = Population(POPULATION_COUNT, MUTATION_RATE, [in_dimen, 13, 8, 13, out_dimen])
# pop = Population(POPULATION_COUNT, MUTATION_RATE, [in_dimen, 2, 3, 2, out_dimen])
# pop = Population(POPULATION_COUNT, MUTATION_RATE, [in_dimen, 4, 3, 4, out_dimen])
bestNeuralNets = []

# print("\nObservation\n--------------------------------")
# print("Shape :", in_dimen, " \n High :", obsMax, " \n Low :", obsMin)
# print("\nAction\n--------------------------------")
# print("Shape :", out_dimen, " | High :", actionMax, " | Low :", actionMin,"\n")

for gen in range(MAX_GENERATIONS):
    genAvgFit = 0.0
    minFit =  1000000
    maxFit = -1000000
    max_steps = 0
    maxNeuralNet = None
    reward_map = {}
    step_map = {}
    availiable_opponents = list(range(len(pop.population)))
    random.shuffle(availiable_opponents)
    for pop_i, nn in tqdm.tqdm(list(enumerate(pop.population))):
        avg_steps = []
        for i in range(COMPARISON_TESTS):
            if len(reward_map.get(pop_i, [])) >= COMPARISON_TESTS:
                continue
            opponent_i = None
            if len(availiable_opponents) > i:
                opponent_i = availiable_opponents[-1*(i+1)]
                opponent_tests = len(reward_map.get(opponent_i, []))
                while opponent_tests >= COMPARISON_TESTS and len(availiable_opponents) > i+1:
                    availiable_opponents.pop(-1*(i+1))
                    opponent_i = availiable_opponents[-1*(i+1)]
                    opponent_tests = len(reward_map.get(opponent_i, []))
            if len(availiable_opponents) <= i+1 or opponent_i is None:
                opponent_i = np.random.choice(range(len(pop.population)))
            opponent = pop.population[opponent_i]
            ob_l, ob_r = env.reset()
            totalReward_l = 0
            totalReward_r = 0
            action_l = None
            action_r = None
            for step in range(MAX_STEPS):
                #env.render()
                # action = 0
                if step % 10 == 0:
                    action_l = opponent.getOutput(ob_l)
                    action_r = nn.getOutput(ob_r)
                
                # action = argmax(nn.getOutput(observation))
                (ob_l, ob_r), (r_l, r_r), done, info = env.step(action_l, action_r)
                # totalReward += reward
                totalReward_l += r_l
                totalReward_r += r_r
                if done:
                    break
            avg_steps.append(step)
            # print(totalReward)
            r_rewards = reward_map.setdefault(pop_i, [])
            r_rewards.append(totalReward_r)
            l_rewards = reward_map.setdefault(opponent_i, [])
            l_rewards.append(totalReward_r)
    for pop_i, nn in enumerate(pop.population):
        rewards = reward_map[pop_i]
        totalReward = np.mean(rewards)
        avg_steps = np.mean(avg_steps)
        nn.fitness = totalReward
        minFit = min(minFit, nn.fitness)
        genAvgFit += nn.fitness
        if nn.fitness > maxFit :
            maxFit = nn.fitness
            max_steps = avg_steps
            maxNeuralNet = copy.deepcopy(nn);
    # with open("bestnn.json", "wb+") as f:
    #     pickle.dump(maxNeuralNet, f)
    maxNeuralNet.to_json("bestnn.json")
    
    bestNeuralNets.append(maxNeuralNet)
    genAvgFit/=pop.popCount
    print("Generation : %3d  |  Min : %5.0f  |  Avg : %5.0f  |  Max : %5.0f  | Steps: %5.0f" % (gen+1, minFit, genAvgFit, maxFit, max_steps) )
    pop.createNewGeneration(maxNeuralNet)

input("Press a key to show best NN")
recordBestBots(bestNeuralNets)
# show_video()