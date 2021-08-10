
import numpy as np
import random, bisect, copy
from neural_network import NeuralNet, FastNeuralNet


def try_mutate_value(mutated_value, m_rate, parent1_value, parent2_value, parent1_fitness, total_fitness):
    """ A helper function while takes a random mutated value, as well as values from two parents, and randomly selects which to return.
    The chance of returning each value is based on a mutation rate, as well as the fitness of each parent, i.e., the better parent is more likely
    to pass on its genes.
    """
    # Check if a mutation occured
    mutation_roll = np.random.rand()
    mutation_occured = mutation_roll < m_rate
    if mutation_occured:
        return mutated_value

    # If a mutation did not occur, We need to inherit a value from a parent
    # We pick the parent based on the fitness of each parent,
    # i.e., a parent with a higher fitness is more likely to pass
    # on its value
    inherit_from_p1_roll = np.random.rand()
    p1_proportion_of_fitness = parent1_fitness / total_fitness
    inherit_from_p1 = inherit_from_p1_roll < p1_proportion_of_fitness
    if inherit_from_p1:
        child_value = parent1_value
    else:
        child_value = parent2_value
    return child_value

def createChild(nn1, nn2, mutation_rate):
    """ Create a child neural network by combining two NeuralNets
    """
    # Create a new completely random neural network
    # Each parameter here represents a 'mutation'
    child = NeuralNet(nn1.nodeCount)

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

        bias = try_mutate_value(child_neuron.bias, mutation_rate, parent1_value=parent1_neuron.bias, parent2_value=parent2_neuron.bias, parent1_fitness=nn1.fitness, total_fitness=total_fitness)
        child_neuron.set_bias(bias)

        parent1_weights = parent1_neuron.connection_weights
        parent2_weights = parent2_neuron.connection_weights
        for w, mutated_weight in enumerate(child_neuron.connection_weights):
            weight = try_mutate_value(mutated_weight,mutation_rate, parent1_value=parent1_weights[w], parent2_value=parent2_weights[w], parent1_fitness=nn1.fitness, total_fitness=total_fitness)
            child_neuron.connection_weights[w] = weight

    return child

class Population :
    def __init__(self, populationCount, mutationRate, nodeCount):
        """ Generates a population of random neural networks
        Arguments:

          populationCount : int
            - The number of neural networks to generate
          

        """
        self.nodeCount = nodeCount
        self.popCount = populationCount
        self.m_rate = mutationRate

        self.population = [NeuralNet(nodeCount) for i in range(populationCount)]

    def select_random_parent(self, retained_members):
        fitness = [x.fitness for x in retained_members]
        # We need positive fitness to calculate a probability, so we will just make the minimum value 0
        min_fitness = min(fitness)
        positive_fitness = [f+abs(min_fitness) for f in fitness]
        total_fitness = sum(positive_fitness)

        # Give each member a probability based on its fitness
        probabilities = [f/total_fitness for f in positive_fitness]

        return np.random.choice(retained_members, p=probabilities)


    def createNewGeneration(self):    
        nextGen = []
        # Sort out population in terms of fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # We will retain some proportion of current members as they are
        proportion_retained = 0.25
        n_retained = int(len(self.population) * proportion_retained)
        retained_members = []
        for i in range(n_retained):
            copied = copy.deepcopy(self.population[i])
            retained_members.append(copied);
            nextGen.append(copied)
        
        # Now we want to fill the generation with children.
        # We will randomly select
        while(len(nextGen) < self.popCount):
            parent_1 = self.select_random_parent(retained_members)
            parent_2 = self.select_random_parent(retained_members)
            nextGen.append( createChild(parent_1, parent_2, self.m_rate))
        self.population.clear()
        self.population = nextGen

    
    #************************************************
    #********YOU CAN IGNORE THE CODE BELOW***********
    #************************************************
    #************************************************
    def createChildFast(self, nn1, nn2):
        """ Create a child neural network by combining two FastNeuralNets
        """
        child = FastNeuralNet(self.nodeCount)
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


