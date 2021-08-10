import numpy as np
import json


class Neuron:
    """ A simple linear neuron, with no activation function.
    """
    def __init__(self):
        self.connection_weights = []
        self.bias = None
    
    def add_connection(self, weight):
        self.connection_weights.append(weight)
    
    def set_bias(self, bias):
        self.bias = bias
    
    def get_output(self, input_signal):
        if len(input_signal) != len(self.connection_weights):
            raise ValueError("Size of input different to size of connection!")

        # The output signal for a neuron is the sum of the signals received from connected
        # input neurons. Each input signal is multiplied by a weight, you could think of this as
        # the influence of that input. In a real brain, this could represent the conductivity of the nerve connection.
        output_signal = 0
        for input, weight in zip(input_signal, self.connection_weights):
            output_signal += input * weight
        
        # Then we add the bias, some extra bit of output signal this neuron always adds to its output
        output_signal += self.bias

        return output_signal
    
    # The following code just allows saving and loading the model
    # Feel free to ignore!
    def to_json(self):
        data = {
            "bias": self.bias,
            "weights": self.connection_weights
        }
        return json.dumps(data)
    
    def from_json_str(self, json_str):
        json_data = json.loads(json_str)
        self.bias = json_data["bias"]
        self.connection_weights = json_data["weights"]

class NeuralNet:
    """ Represents an 'artificial brain', made up of neurons.
    Simpler and more pythonic, but may run slower than the fast version
    making use of matrix multiplication.
    """
    def __init__(self, nodeCount):
        """Arguments:
            nodeCount: List[int] 
                - A list of ints representing the sizes of each layer of the neural network.
                The first value should be the size of the input layer, and the last value should be the size
                of the output layer. The rest is up to you! More layers, and wider layers, can represent more
                complex interactions, but may also need more time to train because there are more numbers to 
                select and get right. 
            seed: int
                - Controls the random initialization of parameters. Setting the same seed should generate the same
                network. Useful for reproducibility, but can leave as None to just generate randomly.
        """

        self.nodeCount = nodeCount

        # Stores the fitness of the neural network.
        # Set externally by some evaluation function    
        self.fitness = 0.0

        # Generate random neurons to connect in the brain
        self.neuron_layers = []
        for i in range(1, len(nodeCount)):
            neuron_layer = []
            layer_n_neurons = self.nodeCount[i]
            for neuron_index in range(layer_n_neurons):
                # Create our new neuron
                neuron = Neuron()

                # Set a random bias
                neuron.set_bias(np.random.uniform(low=-1, high=1))

                # Make a connection to each neuron in the previous layer
                # with a random strength
                prev_n_neurons = self.nodeCount[i-1]
                for prev_index in range(prev_n_neurons):
                    connection_weight = np.random.uniform(low=-1, high=1)
                    neuron.add_connection(connection_weight)
                neuron_layer.append(neuron)
            self.neuron_layers.append(neuron_layer)
  
    def getOutput(self, input):
        """ Get the output of the neural network given some input
        """
        layer_input = input

        # Output is calculated layer by layer.
        # We calculate the output of the first layer of neurons given the input.
        # We then feed the output of the first layer as input to the second.
        # Repeats until we have gone through all layers, and we can just return.
        for layer in self.neuron_layers:
            layer_output = []
            for neuron in layer:
                neuron_output = neuron.get_output(layer_input)
                layer_output.append(neuron_output)
            # Set output to be input to the next layer
            layer_input = layer_output
        return layer_output
    

    # The following code just allows saving and loading the model
    # Feel free to ignore!
    def to_json(self, fn):
        nn_data = {
            "fitness": self.fitness,
            "nodeCount": self.nodeCount,
            "neurons": [[n.to_json() for n in layer] for layer in self.neuron_layers],
        }
        with open(fn, 'w+') as f:
            json.dump(nn_data, f)
    
    def from_json(self, fn):
        with open(fn, 'r') as f:
            nn_data = json.load(f)
        self.from_json_data(nn_data)

    def from_json_str(self, json_str):
        nn_data = json.loads(json_str)
        self.from_json_data(nn_data)

    def from_json_data(self, nn_data):
        self.fitness = nn_data["fitness"]
        self.nodeCount = nn_data["nodeCount"]
        nn_layers = nn_data["neurons"]
        self.neuron_layers = []
        for layer in nn_layers:
            nn_layer = []
            for n in layer:
                neuron = Neuron()
                neuron.from_json_str(n)
                nn_layer.append(neuron)
            self.neuron_layers.append(nn_layer)
        



#************************************************
#********YOU CAN IGNORE THE CODE BELOW***********
#************************************************
#************************************************
class FastNeuralNet:
    """ Represents an 'artificial brain', made up of neurons
    Rather than representing neurons individually, we just keep a matrix of weights and biases.
    We can quickly calculate the same output using matrix multiplication.
    """
    def __init__(self, nodeCount):     
        self.fitness = 0.0
        self.nodeCount = nodeCount
        self.weights = []
        self.biases = []
        for i in range(len(nodeCount) - 1):
            self.weights.append( np.random.uniform(low=-1, high=1, size=(nodeCount[i], nodeCount[i+1])).tolist() )
            self.biases.append( np.random.uniform(low=-1, high=1, size=(nodeCount[i+1])).tolist())


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
            output = np.reshape(np.matmul(output, self.weights[i]) + self.biases[i], (self.nodeCount[i+1]))
        return output
    
    def to_json(self, fn):
        nn_data = {
        "fitness": self.fitness,
            "nodeCount": self.nodeCount,
            "weights": self.weights,
            "biases": self.biases,
        }
        with open(fn, 'w+') as f:
            json.dump(nn_data, f)
    
    def from_json(self, fn):
        with open(fn, 'r') as f:
            nn_data = json.load(fn)
        self.from_json_data(nn_data)

    def from_json_str(self, json_str):
        nn_data = json.loads(json_str)
        self.from_json_data(nn_data)
    
    def from_json_data(self, nn_data):
        self.fitness = nn_data["fitness"]
        self.nodeCount = nn_data["nodeCount"]
        self.weights = nn_data["weights"]
        self.biases = nn_data["biases"]
