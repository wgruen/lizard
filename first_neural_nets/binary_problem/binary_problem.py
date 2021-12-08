#!/usr/bin/env python3                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              #!/usr/bin/env python3

import sys
import argparse
import os
import inspect
import importlib
from numpy import exp, array, random, dot, round
import numpy as np
from os import linesep
import pprint, pickle
import json
import yaml
import decimal



'''
Set the path

Pretty tricky to run this in Pythonista on an iPad
'''
print("pwd: " + os.getcwd())
#os.chdir(os.getcwd())

# use this if you want to include modules from a subfolder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0])))
#print("new sub folder: " + cmd_subfolder)
#print("pardir: " + os.pardir)

sys.path.insert(0, "..")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


if cmd_subfolder not in sys.path:
    #print("new sub folder: " + cmd_subfolder)
    sys.path.insert(0, cmd_subfolder)
#print(sys.path)

from nr_lib import nr_1_hidden_layer as nr


'''
Train a machine and store the machine as a binary blob
'''
def learn(configuration):
    # Load the training data
    file_name = configuration["training_data"]["training_data_file_name"]
    print("Input file name: "+ file_name)
    
    training_set_inputs = np.loadtxt(file_name)
    print("Raw training_set_inputs: ****************" + linesep + str(training_set_inputs))
    # extract input data and expected output
    training_set_outputs = training_set_inputs[:, -1] # for last column
    training_set_inputs =  training_set_inputs[:, :-1] # for all but last column
    # Tur output into proper format
    training_set_outputs = training_set_outputs.reshape(training_set_outputs.shape[0], 1)
    
    print("training_set_inputs: ****************"  + linesep + str(training_set_inputs))
    print("training_set_outputs: ****************" + linesep + str(training_set_outputs))
    
    
    #Intialise a single neuron neural network.
    # currently only on hidden layer is supported
    number_of_input_neurons   = configuration["training_data"]["number_of_input_neurons"]
    number_of_hidden_neurons  = configuration["training_data"]["number_of_hidden_neurons"]
    number_of_hidden_layers   = configuration["training_data"]["number_of_hidden_layers"]
    number_of_outputs_neurons = configuration["training_data"]["number_of_outputs_neurons"]
    
    # create the neural network
    neural_network = nr.NeuralNetwork(
            number_of_input_neurons,
            number_of_hidden_neurons, 
            number_of_hidden_layers, 
            number_of_outputs_neurons)

    
  
    print("synaptic weights starting random values ****************:", linesep)
    print(neural_network.show_synaptic_weights(), linesep)
      
    #sys.stdout = open(os.devnull, 'w')

    # Train the neural network
    number_of_training_iterations = configuration["training_data"]["number_of_training_iterations"]
    neural_network.train(training_set_inputs, training_set_outputs, number_of_training_iterations)
    #sys.stdout = sys.__stdout_
    

    print("synaptic weights after training ****************:", linesep)
    print(neural_network.show_synaptic_weights(), linesep)
    
    print("All matrixes **************** Start: ")
    print(neural_network.show_matrices())
    print("All matrixes **************** End: ", linesep)
    
    
    # all we need to re-create the network - 
    # is the neural network :-)
    dump_nr_file_name = configuration["machine_dump_file_name"]
    with open(dump_nr_file_name, 'bw+') as outfile:
        pickle.dump(configuration, outfile)
        pickle.dump(neural_network, outfile)
    
    
'''
Use a previously trained machine with a new data set
'''
def calculate(configuration):
    print("In calculate ****************", linesep)
  
    # Read in theneural machine, wich was used to with the training data
    # No need to re-create the machine :-)
    dump_nr_file_name = configuration["machine_dump_file_name"]
    with open(dump_nr_file_name,"rb") as pickle_in:
        conifguration  = pickle.load(pickle_in) 
        neural_network = pickle.load(pickle_in)
    
    print("neural_network descrition start **************** ", linesep) 
    print(yaml.dump(configuration, indent=4,  default_flow_style=False, default_style=''))
    pprint.pprint(neural_network)
    print(linesep, "neural_network description end   **************** ", linesep) 
 
    
    test_data = np.loadtxt(configuration["validation_data"])
    test_data =  test_data[:, :-1] # for all but last column

    test_data = np.array(test_data, dtype=np.dtype(decimal.Decimal))
     
     
    print("test data: ****************" + linesep + str(test_data))
   
    print("Do calculaton with the trained network ****************", linesep)
    for data in test_data:
        print("Considering new situation", data, "-> ",  neural_network.think(data))
    
    
'''
The parameters are self explaining

'''
def main(argv):
    parser = argparse.ArgumentParser(description='Training and running a small neural network.')
    parser.add_argument('-i', dest='input_config_file', 
                        action='store', 
                        required=True,
                        help='The yaml file containing configuration')
    
    parser.add_argument('-t', dest='train', action='store', 
                        required=False,
                        default=False,
                        help='train the machine')
    
    parser.add_argument('-r', dest='run', action='store', 
                        required=False,
                        default=False,
                        help='run the machine')
    args = parser.parse_args()
    #print(args)
       
       
    configuration = None
    with open(args.input_config_file, 'r') as stream:
        configuration = yaml.safe_load(stream)
        
    print(yaml.safe_dump(configuration, default_flow_style=False, default_style=None))

    if args.train:
        learn(configuration)
        
    if args.run:     
        calculate(configuration)
    
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
