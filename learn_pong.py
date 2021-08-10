from gym import logger as gymlogger
from numpy.core.fromnumeric import argmax
gymlogger.set_level(40) #error only
import numpy as np
import random




import random, copy
import numpy as np
import tqdm


from customPong import CustomPong
from genetic_evolution import Population
from utils import displayMultiAgentGame, evaluate_multi_agent_game

def evaluate_population(pop, n_tests=5):

    # We will be playing a few games with each member, so lets create a record of the rewards they earn
    reward_record = {}

    # Initialize records for each member
    for pop_i, nn in list(enumerate(pop.population)):
        reward_record[pop_i] = []

    # We create a random order to handle matchups
    availiable_opponents = list(range(len(pop.population)))
    np.random.shuffle(availiable_opponents)

    # We are going to test each AI n_tests times
    for i in range(n_tests):
        print(f"Test {i}")
        # Loop testing each member of the population
        for pop_i, nn in list(enumerate(pop.population)):

            # lets get the record for games this AI has already played
            member_record = reward_record[pop_i]

            # We may have played all required games as an opponent!
            if len(member_record) >= n_tests:
                continue

            # Select a new opponent, which hasn't already played all its matches
            opponent_i = availiable_opponents[-1]
            opponent = pop.population[opponent_i]
            opponent_record = reward_record[opponent_i]

            reward_member, reward_opponent = evaluate_multi_agent_game(nn, opponent)

            member_record.append(reward_member)
            opponent_record.append(reward_opponent)

            # Take the opponent off availiable_opponents if it has played enough matches
            while len(availiable_opponents) > 0 and len(reward_record[availiable_opponents[-1]]) >= n_tests:
                availiable_opponents.pop()

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

MAX_STEPS = 15000
# Increase this to evolve for more generations!
# Will give better results
MAX_GENERATIONS = 10
POPULATION_COUNT = 50
MUTATION_RATE = 0.02
COMPARISON_TESTS = 10

env = CustomPong()
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
bestNeuralNets = []

for gen in range(MAX_GENERATIONS):
    min_pop_fitness, avg_pop_fitness, max_pop_fitness, best_player = evaluate_population(pop, n_tests=COMPARISON_TESTS)
    # Save the current best AI
    best_player.to_json("saved_models/learned_pong_nn.json")
    print(f"Generation: {gen} | Min: {min_pop_fitness} | Max: {max_pop_fitness} | Avg: {avg_pop_fitness}")

    # Create New Generation!
    pop.createNewGeneration()

# Finally, we pause to view the best NN generated in the final Generation
input("Press a key to show best NN")
displayMultiAgentGame(best_player, best_player, CustomPong)
# show_video()