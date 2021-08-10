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
from utils import displaySingleAgentGame, evaluate_single_agent_game


def evaluate_walker_population(pop, n_tests=5):

    # We will be playing a few games with each member, so lets create a record of the rewards they earn
    reward_record = {}

    # Initialize records for each member
    for pop_i, nn in list(enumerate(pop.population)):
        reward_record[pop_i] = []

    # We are going to test each AI n_tests times
    for i in range(n_tests):
        print(f"Test {i}")
        # Loop testing each member of the population
        for pop_i, nn in list(enumerate(pop.population)):

            # lets get the record for games this AI has already played
            member_record = reward_record[pop_i]

            reward = evaluate_single_agent_game(nn)
            member_record.append(reward)

  
    min_pop_fitness = 100000
    max_pop_fitness = -100000
    avg_pop_fitness = 0
    best_player = None
    # Now lets set the fitness of each AI to be its average score
    for pop_i, nn in enumerate(pop.population):
        rewards = reward_record[pop_i]
        avg_reward = np.mean(rewards)
        nn.fitness = avg_reward

        # And track min, max and avg of the population
        min_pop_fitness = min(min_pop_fitness, nn.fitness)
        avg_pop_fitness += nn.fitness
        if nn.fitness > max_pop_fitness :
            max_pop_fitness = nn.fitness
            best_player = nn
    avg_pop_fitness /= len(pop.population)
    return min_pop_fitness, avg_pop_fitness, max_pop_fitness, best_player

MAX_STEPS = 2500

# Increase this to evolve for more generations!
# Will give better results
MAX_GENERATIONS = 50
POPULATION_COUNT = 50
MUTATION_RATE = 0.02
COMPARISON_TESTS = 1

GAME = 'BipedalWalker-v3'
env = gym.make(GAME)
observation = env.reset()

# Automatically parse the size of input and output for the environment
input_dimensions = env.observation_space.shape[0]
try:
    output_dimensions = env.action_space.shape[0]
except:
    output_dimensions = env.action_space.n


# We first create a population of neural networks.
# For each generation, the neural networks play eachother, recording their performance into reward_map
# Once we have the fitness scores for all NNs, we evolve a new generation. We randomly select some NNs to pass into
# the next generation, with better performing NNs being more likely.
# We then fill the rest of the generation with children, which inherit from randomly selected parents.
pop = Population(POPULATION_COUNT, MUTATION_RATE, [input_dimensions, 13, 8, 13, output_dimensions])

for gen in range(MAX_GENERATIONS):
    min_pop_fitness, avg_pop_fitness, max_pop_fitness, best_player = evaluate_walker_population(pop, n_tests=COMPARISON_TESTS)
    # Save the current best AI
    best_player.to_json(f"saved_models/learned_walker_nn_gen_{gen}.json")
    print(f"Generation: {gen} | Min: {min_pop_fitness} | Max: {max_pop_fitness} | Avg: {avg_pop_fitness}")

    # Create New Generation!
    pop.createNewGeneration()

# Finally, we pause to view the best NN generated in the final Generation
input("Press a key to show best NN")
displaySingleAgentGame(best_player, lambda : gym.make(GAME), max_steps=1500)
# show_video()