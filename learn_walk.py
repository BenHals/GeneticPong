import gym
from gym import logger as gymlogger
from numpy.core.fromnumeric import argmax
gymlogger.set_level(40) #error only
import numpy as np
import random

import random, copy
import numpy as np
import tqdm

from genetic_evolution import Population
from utils import displaySingleAgentGame


GAME = 'BipedalWalker-v3'
MAX_STEPS = 1500
MAX_GENERATIONS = 20
POPULATION_COUNT = 500
MUTATION_RATE = 0.01
nn_type = "slow"
env = gym.make(GAME)
observation = env.reset()

# Automatically parse the size of input and output for the environment
in_dimen = env.observation_space.shape[0]
try:
    out_dimen = env.action_space.shape[0]
except:
    out_dimen = env.action_space.n

# We first create a population of neural networks.
# For each generation, the neural networks play eachother, recording their performance into reward_map
# Once we have the fitness scores for all NNs, we evolve a new generation. We randomly select some NNs to pass into
# the next generation, with better performing NNs being more likely.
# We then fill the rest of the generation with children, which inherit from randomly selected parents.
pop = Population(POPULATION_COUNT, MUTATION_RATE, [in_dimen, 13, 8, 13, out_dimen], neural_net_type=nn_type)
bestNeuralNets = []

for gen in range(MAX_GENERATIONS):
    # Just some monitoring parameters
    avg_generation_fitness = 0.0
    min_generation_fitness =  1000000
    max_generation_fitness = -1000000
    avg_generation_steps = 0
    maxNeuralNet = None
    avg_steps = []
    rewards = []

    # Loop testing each member of the population
    for pop_i, nn in tqdm.tqdm(list(enumerate(pop.population))):

        # Setup the game
        ob = env.reset()
        totalReward = 0
        action = None
        for step in range(MAX_STEPS):
            action = nn.getOutput(ob)
            
            ob, reward, done, info = env.step(action)
            totalReward += reward
            if done:
                break
        avg_steps.append(step)
        rewards.append(totalReward)

    # We evaluate the results of all matches, and set the fitness scores of the 
    # NNs
    # And pick the best one from this generation to save to disk
    for pop_i, nn in enumerate(pop.population):
        totalReward = rewards[pop_i]
        nn.fitness = totalReward
        min_generation_fitness = min(min_generation_fitness, nn.fitness)
        avg_generation_fitness += nn.fitness
        if nn.fitness > max_generation_fitness :
            max_generation_fitness = nn.fitness
            maxNeuralNet = copy.deepcopy(nn)
    maxNeuralNet.to_json(f"saved_models/learned_walker_nn_gen_{gen}.json")
    bestNeuralNets.append(maxNeuralNet)

    # Monitoring ourput
    avg_steps = np.mean(avg_steps)
    avg_generation_steps = avg_steps
    avg_generation_fitness/=pop.popCount
    print("Generation : %3d  |  Min : %5.0f  |  Avg : %5.0f  |  Max : %5.0f  | Steps: %5.0f" % (gen+1, min_generation_fitness, avg_generation_fitness, max_generation_fitness, avg_generation_steps) )

    # Create New Generation!
    pop.createNewGeneration(maxNeuralNet)

# Finally, we pause to view the best NN generated in the final Generation
input("Press a key to show best NN")
displaySingleAgentGame(bestNeuralNets[-1], lambda : gym.make(GAME), max_steps=1500)
# show_video()