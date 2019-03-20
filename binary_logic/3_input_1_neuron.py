# -*- coding: utf-8 -*-
"""

This learning problem has one neuron, with three input variables

This engine is limited and produces at least one error.
Also for the input of [0,0,0], it does out 0.5 consistently. 
But in this case, the dot operation results in zero, so this may be ok with this simple machine.

"""
#!/usr/bin/env python3

from numpy import exp, array, random, dot, round
from os import linesep


class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1


    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    # https://en.wikipedia.org/wiki/Sigmoid_function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))


    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)


    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):

        print("Learning inputs:", linesep, training_set_inputs, linesep)
        print("Learning expected outputs:", linesep, training_set_outputs, linesep)
        
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think_brain(training_set_inputs)
            #print("Learning output:", linesep, output, linesep)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output

            #print("Learning error:", linesep, error, linesep)


            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            #print("Learning adjustment:", linesep, adjustment, linesep)
            #print("Adustments:", linesep, adjustment, linesep)

            # Adjust the weights.
            self.synaptic_weights += adjustment
            #break


    # The neural network thinks.
    def think_brain(self, inputs):
        # Pass inputs through our neural network (our single neuron).
        # use the Dot product
        # https://en.wikipedia.org/wiki/Dot_product
        
        dot_product = dot(inputs, self.synaptic_weights)
        print("Dot product:", linesep, dot_product, linesep)
        
        after_sigmoid = self.__sigmoid(dot_product)
        print("After sigmoid:", linesep, after_sigmoid, linesep)
        return after_sigmoid


    # The neural network thinks.
    def think(self, inputs):
        thinking_results = self.think_brain(inputs)
        return thinking_results, round(thinking_results)
        

if __name__ == "__main__":

    # The training set. We have 5 examples, each consisting of 3 input values
    # and 1 expectd output value.
    training_set_inputs = array([[0, 0, 1], 
                                 [1, 1, 1],
                                 [1, 0, 1],
                                 [1, 1, 1]])
    
    training_set_outputs = array([[0,
                                   1,
                                   0,
                                   1]]).T
    
    
    #Intialise a single neuron neural network.
    neural_network = NeuralNetwork()
    print("Random starting synaptic weights:", linesep, neural_network.synaptic_weights, linesep)

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training:", linesep, neural_network.synaptic_weights, linesep)

    # Test the neural network with a new situation.
    test_data = array([[0, 0, 0], 
                       [0, 0, 1],
                       [0, 1, 0],
                       [0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]])
    
    for data in test_data:
        print("Considering new situation", data, "-> ",  neural_network.think(data))
    


    
    
    
    
