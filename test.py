from neural_network import Neuron, FastNeuralNet, NeuralNet
from utils import show_neuron, show_network


neuron = Neuron()
neuron.set_bias(5)
neuron.add_connection(3)
neuron.add_connection(2)
neuron.add_connection(1)

# show_neuron(neuron, [3, 4, 2])

nn = NeuralNet([3, 4, 2, 4, 3])
print(nn.getOutput([3, 4, 2]))
show_network(nn, [3, 4, 2])