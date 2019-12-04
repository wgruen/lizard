#!/usr/bin/env python3

import sys
import os
import inspect
import importlib
from numpy import exp, array, random, dot, round
import numpy as np
from os import linesep
import pprint, pickle

print("pwd: " + os.getcwd())
#os.chdir(os.getcwd())

# use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0])))
print("new sub folder: " + cmd_subfolder)
print("pardir: " + os.pardir)

sys.path.insert(0, "..")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



if cmd_subfolder not in sys.path:
    print("new sub folder: " + cmd_subfolder)
    sys.path.insert(0, cmd_subfolder)
print(sys.path)



from nr_lib import nr_1_hidden_layer as nr


def learn(dump_nr_file_name):
    
    
    # The training set. We have 4 examples, each consisting of 3 input values
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
    number_of_input_neurons = 3
    number_of_hidden_neurons = 4
    number_of_hidden_layers = 1
    number_of_outputs_neurons = 1
    neural_network = nr.NeuralNetwork(
            number_of_input_neurons,
            number_of_hidden_neurons, 
            number_of_hidden_layers, 
            number_of_outputs_neurons)

    
  
    print("synaptic weights Random ****************:", linesep)
    print(neural_network.show_synaptic_weights(), linesep)
    
    
    #sys.stdout = open(os.devnull, 'w')

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 1000)
    #sys.stdout = sys.__stdout_
    

    print("synaptic weights after training ****************:", linesep)
    print(neural_network.show_synaptic_weights(), linesep)
    
    print("All matrixes **************** Start: ")
    print(neural_network.show_matrices())
    print("All matrixes **************** End: ", linesep)
    
    
    # all we need to re-create the network
    # is the neural network :-)
    with open(dump_nr_file_name, 'bw+') as outfile:
        pickle.dump(neural_network, outfile)
        
        
        a_len = len(neural_network.neuron_weigths_holder)
        for i in range(0, a_len):
            a2_len = len(neural_network.neuron_weigths_holder[i])
            a = np.ndarray(a2_len, buffer=neural_network.neuron_weigths_holder[i])
            print(a)
            #a.dump(outfile)
            #pickle.dump(a.dump, outfile)
            pickle.dump(neural_network.neuron_weigths_holder[i], outfile)
   


def calculate(dump_nr_file_name):
    print("In calculate ****************", linesep)
    #read the pickle file
    pickle_in = open(dump_nr_file_name,"rb")
    pickle_data = pickle.load(pickle_in)
    
    print("pickle data start **************** ", linesep) 
    pprint.pprint(pickle_data)
    print("pickle data end   **************** ", linesep) 
 
    

    # Test the neural network with a new situation.
    test_data = array([[0, 0, 0], 
                       [0, 0, 1],
                       [0, 1, 0],
                       [0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 1],
                       [1, 1, 0],
                       [1, 1, 1]])
   
    print("Do calculaton with the trained network ****************", linesep)
    for data in test_data:
        print("Considering new situation", data, "-> ",  pickle_data.think(data))
    
    
def main():
    dump_nr_file_name = "binary_problem_trained.bin"
    learn(dump_nr_file_name)
    calculate(dump_nr_file_name)
    
    
    
if __name__ == '__main__':
    main()
    
    
    
