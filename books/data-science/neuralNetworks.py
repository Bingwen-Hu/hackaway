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

""" again: weights are the key! so we need a method to automatically define 
weights for us -- that's feed-forward neural networks """

""" is sigmoid same as logistic?  sigmoid refers to the shape of the function
Note that sigmoid is continous so can be derivated
"""
def sigmoid(t):
    return 1 / (1 + np.exp(-t))

def neuron_output(weights, inputs):
    return sigmoid(np.dot(weights, inputs))




