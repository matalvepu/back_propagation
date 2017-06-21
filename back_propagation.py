import numpy as np
from random import seed
from random import random

# initialize network


def initialize_network(network_config):
    network = list()
    for i in range(1, len(network_config)):
        layer = [{'weights': [random() for j in range(network_config[i-1] + 1)],
                  'delta':None, 'output':None} for j in range(network_config[i])]
        network.append(layer)
    return network

#*****************************forward propagation*************************

# neuron activation
'''
Neuron activation is calculated as the weighted sum of the inputs. Much like linear regression.

activation = sum(weight_i * input_i) + bias
'''

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# neuron transfer

'''
Once a neuron is activated, we need to transfer the activation to see what the neuron 
output actually is.Different transfer functions can be used. It is traditional to use 
the sigmoid activation function.

Other functions:
1.hyperbolic tangent
2.rectifier transfer function
'''
def transfer(activation, f):
    return f(activation)


# propagate the input to output
'''
Forward propagating an input is straightforward.
We work through each layer of our network calculating the outputs for each neuron. 
All of the outputs from one layer become inputs to the neurons on the next layer.
Below is a function named forward_propagate() that implements the forward propagation 
for a row of data from our dataset with our neural network.You can see that a neurons 
output value is stored in the neuron with the name output. You can also see that we 
collect the outputs for a layer in an array named new_inputs that becomes the array 
inputs and is used as inputs for the following layer.The function returns the outputs 
from the last layer also called the output layer.
'''
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation, sigmoid)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    outputs = inputs
    return outputs


#*************************Back Propagate Error***************************************

# Calculate the derivative of an neuron output
#for sigmoid function: derivative = output * (1.0 - output)
def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store the errors in neurons
'''
The first step is to calculate the error for each output neuron, this 
will give us our error signal (input) to propagate backwards through the network.
The error for a given neuron can be calculated as follows:

	error = (expected - output) * transfer_derivative(output)

Where expected is the expected output value for the neuron, output is the output 
value for the neuron and transfer_derivative() calculates the slope of the neurons 
output value, as shown above. This error calculation is used for neurons in the 
output layer. The expected value is the class value itself. 

In the hidden layer, things are a little more complicated.The error signal for a 
neuron in the hidden layer is calculated as the weighted error of each neuron in
the output layer.Think of the error traveling back along the weights of the output 
layer to the neurons in the hidden layer.The back-propagated error signal is accumulated and 
then used to determine the error for the neuron in the hidden layer, as follows:

	error = (weight_k * error_j) * transfer_derivative(output)

Where error_j is the error signal from the jth neuron in the output layer,
weight_k is the weight that connects the kth neuron to the current neuron and 
output is the output for the current neuron.
'''

def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
'''
Once errors are calculated for each neuron in the network via the back propagation
method above, they can be used to update weights. Network weights are updated as 
follows:

	weight = weight + learning_rate * error * input

Where weight is a given weight, learning_rate is a parameter that you must specify,
error is the error calculated by the backpropagation procedure for the neuron and 
input is the input value that caused the error.

The same procedure can be used for updating the bias weight, except there is no 
input term, or input is the fixed value of 1.0.
'''
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1]-1)] = 1
            sum_error += sum([(expected[i]-outputs[i]) **
                              2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

def evaluate_algorithm(network,test_set):
	predictions=list()
	for row in test_set:
		predictions.append(predict(network,row))

	actual = [int(row[-1]-1) for row in test_set]
	accuracy=accuracy_metric(actual, predictions)
	return accuracy



def load_data():
	file_name="seeds_dataset.txt"
	data=np.loadtxt(file_name).tolist()
	train_length=int(len(data)*0.90)
	train=data[:train_length]
	test=data[train_length:]
	return [train,test]


seed(1)
dataset = load_data()
train_set=dataset[0]
test_set=dataset[1]
n_inputs = len(train_set[0]) - 1
n_outputs = len(set([row[-1] for row in train_set]))

network = initialize_network([n_inputs, 4,3, n_outputs])
print(network)
accuracy=evaluate_algorithm(network,test_set)
print(accuracy)
train_network(network, train_set, 0.5, 50, n_outputs)
print(network)
accuracy=evaluate_algorithm(network,test_set)
print(accuracy)
