
import numpy as np
import random, bisect, copy
from neural_network import SlowNeuralNet, FastNeuralNet


class Population :
    def __init__(self, populationCount, mutationRate, nodeCount, seed = None, neural_net_type="slow"):
        self.nodeCount = nodeCount
        self.popCount = populationCount
        self.m_rate = mutationRate
        if seed is None:
            seed = np.random.randint(10000000)
        self.seed = seed
        # Set up the random number generator used to make parameters
        self.rng = np.random.RandomState(seed)

        self.neural_net_class = SlowNeuralNet if neural_net_type == "slow" else FastNeuralNet
        self.population = [ self.neural_net_class(nodeCount, np.random.randint(10000)) for i in range(populationCount)]
        self.neural_net_type = neural_net_type
        self.createChild = self.createChildSlow if neural_net_type == "slow" else self.createChildFast
    
    def try_mutate_value(self, mutated_value, parent1_value, parent2_value, parent1_fitness, total_fitness):
        """ Get a value for a child node based on inheriting from parents, as well as mutation chance
        """
        mutation_roll = self.rng.rand()
        mutation_occured = mutation_roll < self.m_rate

        if mutation_occured:
            child_value = mutated_value
        else:
            # We need to inherit a value from a parent
            # We pick the parent based on the fitness of each parent,
            # i.e., a parent with a higher fitness is more likely to pass
            # on its value
            inherit_from_p1_roll = self.rng.rand()
            p1_proportion_of_fitness = parent1_fitness / total_fitness
            inherit_from_p1 = inherit_from_p1_roll < p1_proportion_of_fitness
            if inherit_from_p1:
                child_value = parent1_value
            else:
                child_value = parent2_value
        return child_value


    def createChildSlow(self, nn1, nn2):
        """ Create a child neural network by combining two SlowNeuralNets
        """
        # Create a new completely random neural network
        # Each parameter here represents a 'mutation'
        child = self.neural_net_class(self.nodeCount, np.random.randint(10000))

        # Now we are going to adjust some of the neurons to inherit from the parents
        total_fitness = nn1.fitness + nn2.fitness
        all_child_neurons = [n for layer in child.neuron_layers for n in layer]
        all_parent1_neurons = [n for layer in nn1.neuron_layers for n in layer]
        all_parent2_neurons = [n for layer in nn2.neuron_layers for n in layer]

        
        for i, child_neuron in enumerate(all_child_neurons):
            parent1_neuron = all_parent1_neurons[i]
            parent2_neuron = all_parent2_neurons[i]
            # The current data in the child is completely random, i.e., represents 'mutated' data
            # We want to keep this randomness with probability of mutation_rate
            # For bias and each connection weight, we either keep the mutated value or inherit from a parent.

            bias = self.try_mutate_value(child_neuron.bias, parent1_value=parent1_neuron.bias, parent2_value=parent2_neuron.bias, parent1_fitness=nn1.fitness, total_fitness=total_fitness)
            child_neuron.set_bias(bias)

            parent1_weights = parent1_neuron.connection_weights
            parent2_weights = parent2_neuron.connection_weights
            for w, mutated_weight in enumerate(child_neuron.connection_weights):
                weight = self.try_mutate_value(mutated_weight, parent1_value=parent1_weights[w], parent2_value=parent2_weights[w], parent1_fitness=nn1.fitness, total_fitness=total_fitness)
                child_neuron.connection_weights[w] = weight

        return child

    def createChildFast(self, nn1, nn2):
        """ Create a child neural network by combining two FastNeuralNets
        """
        child = self.neural_net_class(self.nodeCount, np.random.randint(10000))
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