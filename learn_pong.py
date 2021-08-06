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
from utils import displayMultiAgentGame


MAX_STEPS = 15000
MAX_GENERATIONS = 200
POPULATION_COUNT = 100
MUTATION_RATE = 0.02
COMPARISON_TESTS = 10
nn_type = "slow"
env = CustomPong()
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


    reward_map = {}

    # We create a random order to handle matchups
    availiable_opponents = list(range(len(pop.population)))
    random.shuffle(availiable_opponents)

    # Loop testing each member of the population
    for pop_i, nn in tqdm.tqdm(list(enumerate(pop.population))):
        # We test each member against some opponents, and take the average score
        for i in range(COMPARISON_TESTS):
            # We can skip if we already have enough tests for this NN
            # (from playing as the opponent for other tests)
            if len(reward_map.get(pop_i, [])) >= COMPARISON_TESTS:
                continue
            
            # Select a new opponent, which hasn't already played all its matches
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

            # Setup the game
            ob_l, ob_r = env.reset()
            totalReward_l = 0
            totalReward_r = 0
            action_l = None
            action_r = None
            for step in range(MAX_STEPS):
                # We only let agents select an action every 10th frame
                # to simulate some kind of reaction delay
                if step % 10 == 0:
                    action_l = opponent.getOutput(ob_l)
                    action_r = nn.getOutput(ob_r)
                
                (ob_l, ob_r), (r_l, r_r), done, info = env.step(action_l, action_r)
                totalReward_l += r_l
                totalReward_r += r_r
                if done:
                    break
            avg_steps.append(step)
            r_rewards = reward_map.setdefault(pop_i, [])
            r_rewards.append(totalReward_r)
            l_rewards = reward_map.setdefault(opponent_i, [])
            l_rewards.append(totalReward_r)

    # We evaluate the results of all matches, and set the fitness scores of the 
    # NNs
    # And pick the best one from this generation to save to disk
    for pop_i, nn in enumerate(pop.population):
        rewards = reward_map[pop_i]
        totalReward = np.mean(rewards)
        nn.fitness = totalReward
        min_generation_fitness = min(min_generation_fitness, nn.fitness)
        avg_generation_fitness += nn.fitness
        if nn.fitness > max_generation_fitness :
            max_generation_fitness = nn.fitness
            maxNeuralNet = copy.deepcopy(nn)
    maxNeuralNet.to_json("saved_models/learned_pong_nn.json")
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
displayMultiAgentGame(bestNeuralNets[-1], bestNeuralNets[-1], CustomPong)
# show_video()