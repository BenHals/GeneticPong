import gym
from gym import logger as gymlogger
from gym.wrappers import Monitor
from numpy.core.fromnumeric import argmax
gymlogger.set_level(40) #error only
# import tensorflow as tf
import numpy as np
import random




import random, copy
import gym
import numpy as np
import tqdm


from customPong import CustomPong
from genetic_evolution import Population
from utils import displayMultiAgentGame


GAME = 'BipedalWalker-v3'
MAX_STEPS = 15000
MAX_GENERATIONS = 20
POPULATION_COUNT = 100
MUTATION_RATE = 0.02
COMPARISON_TESTS = 10
nn_type = "slow"
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
pop = Population(POPULATION_COUNT, MUTATION_RATE, [in_dimen, 13, 8, 13, out_dimen], neural_net_type=nn_type)
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
    availiable_opponents = list(range(len(pop.population)))
    random.shuffle(availiable_opponents)
    avg_steps = []
    for pop_i, nn in tqdm.tqdm(list(enumerate(pop.population))):
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
    maxNeuralNet.to_json("saved_models/bestnn.json")
    
    bestNeuralNets.append(maxNeuralNet)
    genAvgFit/=pop.popCount
    print("Generation : %3d  |  Min : %5.0f  |  Avg : %5.0f  |  Max : %5.0f  | Steps: %5.0f" % (gen+1, minFit, genAvgFit, maxFit, max_steps) )
    pop.createNewGeneration(maxNeuralNet)

input("Press a key to show best NN")
displayMultiAgentGame(bestNeuralNets[-1], bestNeuralNets[-1], CustomPong)
# show_video()