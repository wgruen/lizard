#!/usr/bin/env python3

import sys
import os
import inspect
import importlib
from numpy import exp, array, random, dot, round
import numpy as np
from os import linesep


# use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"lib")))
if cmd_subfolder not in sys.path:
    sys.path.append(cmd_subfolder)
    
    
sys.path.append('.')
    
print(sys.path)

print(cmd_subfolder)




from nr_1_hidden_layer import NeuralNetwork



if __name__ == "__main__":

    # The training set. We have 5 examples, each consisting of 3 input values
    # and 1 expectd output value.
    training_set_inputs = array([[0, 0, 1], 
                                 [1, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 1]])
    
   
    training_set_outputs = array([[0.01,
                                   0.99,
                                   0.01,
                                   0.99]]).T
    
    
    #Intialise a single neuron neural network.
    #number_of_input_neurons, number_of_hidden_neurons, number_of_hidden_layers, number_of_outputs)
    # cueently only on hidden layer is supported
    neural_network = NeuralNetwork(3, 4, 1, 1)

    
  
    print("Random starting synaptic weights:", linesep)
    print(neural_network.show_synaptic_weights(), linesep)
    
    
    #sys.stdout = open(os.devnull, 'w')

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 1000)
    #sys.stdout = sys.__stdout_
    

    print("New synaptic weights after training:", linesep)
    print(neural_network.show_synaptic_weights(), linesep)
    print("All matrixes: ", neural_network.show_matrices())
    
    '''
    with open("neural_net_trained", 'bw+') as outfile:
        pickle.dump(neural_network.training_rate, outfile)
        pickle.dump(neural_network.neuron_weigths_holder, outfile)
        
        
        a_len = len(neural_network.neuron_weigths_holder)
        for i in range(0, a_len):
            a2_len = len(neural_network.neuron_weigths_holder[i])
            a = np.ndarray(a2_len, buffer=neural_network.neuron_weigths_holder[i])
            print(a)
            #a.dump(outfile)
            #pickle.dump(a.dump, outfile)
            pickle.dump(neural_network.neuron_weigths_holder[i], outfile)
            
        
    # finish the write and read later
    #read the pickle file
    pickle_in = open("neural_net_trained","rb")
    print("Read pickle data")
    pickle_data = pickle.load(pickle_in)
    print(pickle_data)
    
 '''  
    

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
    


    
    
    
    
