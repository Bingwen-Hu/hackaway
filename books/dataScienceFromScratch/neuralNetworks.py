"""
Neural Networks are mimiced to brain of human. 
Given some input vectors, calculated throught layers one by one and finanly 
give out some results

important method:
Feed-forward neural networks
Backpropagation

NOTE: the weights are the key!
"""

# perceptrons
import numpy as np
from functools import partial
# fire or not
def step_function(x):
    return 1 if x >= 0 else 0

def perceptron_output(weights, bias, x):
    "return 1 if the perceptron fires, 0 if not"
    calculation = np.dot(weights, x) + bias
    return step_function(calculation)

""" again: weights are the key! """
# build a AND gate
and_gate = partial(perceptron_output, [2, 2], -3)
or_gate = partial(perceptron_output, [2, 2], -1)
not_gate = partial(perceptron_output, [-2], 1)


""" is sigmoid same as logistic?  sigmoid refers to the shape of the function
Note that sigmoid is continous so can be derivated
"""
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(np.dot(weights, inputs))


def feed_forward(neural_network, input_vector):
    """ neural_network = [layer1 layer2 layer3 ... layern]
    layer = [neu1 neu2 neu3 ... neun]
    neu = [wei1 wei2 wei3 ... wein] """
    outputs = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias)
                  for neuron in layer]
        outputs.append(output)

        # update the input_vector
        input_vector = output

    return outputs

# for example
xor_network = [# hidden layer
    [[20, 20, -30],
     [20. 20, -10]],
    # output layer
    [[-60, 60, -30]]]

for x in [0, 1]:
    for y in [0, 1]:
        print(x, y, feed_forward(xor_network, [x, y]))[-1]

""" again: weights are the key! so we need a method to automatically define 
weights for us -- that's backpropagate neural networks """

def backpropagate(network, input_vector, targets):

    hidden_outputs, outputs = feed_forward(network, input_vector)

    # the output*(1-output) is from the derivative of sigmoid
    output_deltas = [output * (1-output) * (output - target)
                     for output, target in zip(outputs, targets)]

    # adjust weights for output layer, one neuron at a time
    for i, output_neuron in enumerate(network[-1]):
        # focus on the ith outpus layer neuron
        for j, hidden_output in enumerate(hidden_outputs + [1]):
            output_neuron[j] -= output_deltas[i] * hidden_output
    # back-propagate errors to hidden layer
    hidden_deltas = [hidden_output * (1- hidden_output) *
                     np.dot(output_deltas, [n[i] for n in network[-1]])
                     for i, hidden_output in enumerate(hidden_outputs)]

    # adjust weights for hidden layer (network[0])
    for i, hidden_neuron in enumerate(network[0]):
        for j, input in enumerate(input_vector + [1]):
            hidden_neuron[j] -= hidden_deltas[i] * input
