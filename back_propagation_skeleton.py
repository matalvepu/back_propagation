#Author: Kamrul Hasan
#this skeleton is adopted from this tuitorial: 
#http://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

import numpy as np
from random import seed
from random import random

# initialize network
'''
Whole netowrk is an array of layers and layers are array of neurons. Each neuron is 
dictionary that contains all weights connected with that neuron from previous layer, 
output of neuron and error occurs due to the output of neuron

Each neuron has a set of weights that need to be maintained. One weight 
for each input connection and an additional weight for the bias. We will 
need to store additional properties for a neuron during training, therefore
we will use a dictionary to represent each neuron and store properties by names 
such as "weights" for the weights.

A network is organized into layers. The input layer is really just a row from our
training dataset. Hidden layers are followed by the output layer that has one neuron  
for each class value.
'''

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
    
    #put your code in here

    return activation

# activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# neuron transfer

'''
Once a neuron is activated, we need to transfer the activation to see what the neuron 
output actually is. Different transfer functions can be used. It is traditional to use 
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
For each layer of our network we have to calcualte the outputs for each neuron. 
All of the outputs from one layer become inputs to the neurons on the next layer.

The output of neuron is stored in the neuron with the name "output". We will return the 
output layers values.
'''
def forward_propagate(network, inputs):
    #put your code here

    

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
The error for a given neuron of output layer can be calculated as follows:

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

    #put your code in here
    #for each neuron calculate delta

    return network



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

    #for each neuron update the weight list

    return network


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[int(row[-1])] = 1
            network=backward_propagate_error(network, expected)
            network=update_weights(network, row, l_rate)




def evaluate_algorithm(network,test_set):
    accuracy=0

    #put your code

    return accuracy



def load_data():
    train_set=list()
    test_set=list()

    #PUT CODE HERE
    return [train_set,test_set]

seed(1)
dataset = load_data()
train_set=dataset[0]
test_set=dataset[1]
network_config=[3,2,4,1]

network = initialize_network(network_config)

print(network)

train_network(network, train_set, 0.5, 50,network_config[-1])

accuracy=evaluate_algorithm(network,test_set)
print(accuracy)
