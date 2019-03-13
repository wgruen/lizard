# -*- coding: utf-8 -*-
"""

This learning problem has three neurons, with three input variables and
three hidden neurons

This engine does not produce errors anymore, probably due more neurons.
Also for the input of [0,0,0], it properly identifies it. 

This is not supposed to be perfect, but it seems to work well enough. 

"""
#!/usr/bin/env python3

from numpy import exp, array, random, dot, round
import numpy as np
from os import linesep


class NeuralNetwork():
    def __init__(self, number_of_input_neurons, number_of_hidden_neurons, number_of_hidden_layers, number_of_outputs):
        
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)
        
        # the inut neurons
        self.neuron_input_holder = []
        self.neuron_weigths_holder = []
        self.neuron_value_holder = []
        self.neuron_normalized_value_holder = []
        self.neuron_error_value_holder = []


        
         # the input neurons   
         # only has the normalized out value
         # which equals the input
        layer = np.full(number_of_input_neurons, 999999999)
        self.neuron_input_holder.append(layer) 
               
        weights = np.full(number_of_input_neurons, 999999999)
        self.neuron_weigths_holder.append(weights) 
        
        value = np.full(number_of_input_neurons, 999999999) 
        self.neuron_value_holder.append(value)         
         
        input_layer = np.full(number_of_input_neurons, 7)
        self.neuron_normalized_value_holder.append(input_layer)
        
        
        # the hidden layer neurons
        number_of_input_neurons = len(input_layer)
        
        # 1st hidden layer uses the inout neurons and the number can be different
        layer = np.zeros((number_of_hidden_neurons, number_of_input_neurons))
        self.neuron_input_holder.append(layer) 
               
        weights = np.random.rand(number_of_hidden_neurons, number_of_input_neurons)
        self.neuron_weigths_holder.append(weights) 
        
        value = np.zeros(number_of_hidden_neurons) 
        self.neuron_value_holder.append(value)
        
        normalized_value = np.full(number_of_hidden_neurons, 9)
        self.neuron_normalized_value_holder.append(normalized_value)
        
        # more hidden layers have the same number of neurons
        for i in range(0, number_of_hidden_layers -1):
            layer = np.zeros((number_of_hidden_neurons, number_of_hidden_neurons))
            self.neuron_input_holder.append(layer) 
               
            weights = np.random.rand(number_of_hidden_neurons, number_of_hidden_neurons)
            self.neuron_weigths_holder.append(weights) 
        
            value = np.zeros(number_of_hidden_neurons) 
            self.neuron_value_holder.append(value)
        
            normalized_value = np.full(number_of_hidden_neurons, 9)
            self.neuron_normalized_value_holder.append(normalized_value)
        
        #The output layer
       # the hidden layer neurons
        layer = np.zeros((number_of_outputs, number_of_hidden_neurons))
        self.neuron_input_holder.append(layer) 
               
        weights = np.random.rand(number_of_outputs, number_of_hidden_neurons)
        self.neuron_weigths_holder.append(weights) 
        
        value = np.zeros(number_of_outputs) 
        self.neuron_value_holder.append(value)
        
        normalized_value = np.ones(number_of_outputs) 
        self.neuron_normalized_value_holder.append(normalized_value)
      
        error_matrix = np.ones(number_of_outputs) 
        self.neuron_error_value_holder.append(error_matrix)
    
        print("Input holder: ", self.neuron_input_holder)
        print("Weights holder: ", self.neuron_weigths_holder)
        print("Value holder:  ", self.neuron_value_holder)
        print("Normalized value holder: ", self.neuron_normalized_value_holder)
        #exit -1
  
        
        

    def show_matrices(self):
        return 
        print("Input holder: ", self.neuron_input_holder)
        print("Weights holder: ", self.neuron_weigths_holder)
        print("Value holder:  ", self.neuron_value_holder)
        print("Normalized value holder: ", self.neuron_normalized_value_holder)

          
           
    def show_synaptic_weights(self):
        print("Weights holder: ", self.neuron_weigths_holder)
    
    
       # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    # https://en.wikipedia.org/wiki/Sigmoid_function
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
  
    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set, training_set_outputs, number_of_training_iterations):
        
        #print("Learning inputs:", linesep, training_set_inputs, linesep)
        #print("Learning inputs size: ", len(training_set_inputs))
        #print("Learning expected outputs:", linesep, training_set_outputs, linesep)
        
        for iteration in range(number_of_training_iterations):
            # walk through the training set
            input_array_size = len(training_set)
            for i in range(input_array_size):
                array = training_set[i]
                expected_output = training_set_outputs[i]
                #print("Expected output: ", expected_output, linesep)
            
                nn_output = self.think(array)
                #print("Neural Network output: ",  nn_output)
                #continue##wolfwolf
                     
                # Calculate the error (The difference between the desired output
                # and the predicted output).
                self.caclulate_the_error(training_set_outputs)
                
                self.caclulate_backward_pass(training_set_outputs)
            
                #break
            #break
            
  #
        
    def think(self, inputs):
        # each input paramter will have its own neuron
        number_of_inputs = len(inputs)
   
        #print("Think Input value: ", inputs)
            
    
        
        # 1st copy the input values into the Input Neurons
        #print(self.neuron_normalized_value_holder[0])
        self.neuron_normalized_value_holder[0] = inputs
        #print(self.neuron_normalized_value_holder[0])
        
        # run the engine
 
        #print(len(self.neuron_normalized_value_holder))
        for layer_index in range(0, len(self.neuron_normalized_value_holder)-1):
            
            #copy the output from the previous layer, as input for the next layer
            out_layer = self.neuron_normalized_value_holder[layer_index]
            in_layer = self.neuron_input_holder[layer_index + 1]
          
    
            #print("layer values out: ", out_layer)
            #print("layer values in: ", in_layer)
            #break
            
            num_of_neruons_in_layer_out = len(out_layer)
            num_of_neruons_in_layer_in = len(in_layer)
            
            #print("Number of neruons in layer OUT: ", num_of_neruons_in_layer_out)
            #print("Number of neurons in layer IN: ", num_of_neruons_in_layer_in, linesep, linesep)
            
            # for each neuron
            for i in range(0, num_of_neruons_in_layer_in):
                # copy the normalized output value from the previous layer into
                # the input values for the next layer
                #print("layer values in before assignment: ", in_layer[i])
            
                in_layer[i] = out_layer
                #print("after values in after assignment: ",  in_layer[i])
                
                # debug        
                self.show_matrices()
                
                # Pass inputs through our neural network (our single neuron).
                # use the Dot product
                # https://en.wikipedia.org/wiki/Dot_product
                
                #print("Neuron Value before dot product: ", self.neuron_value_holder[layer_index + 1][i])
                self.neuron_value_holder[layer_index + 1][i] = \
                         dot(in_layer[i], \
                             self.neuron_weigths_holder[layer_index + 1][i])            
                #print("Neuron Value after dot product: ", self.neuron_value_holder[layer_index + 1][i])

                
                #print("Neuron Normalized Value before sigmoid: ", self.neuron_normalized_value_holder[layer_index + 1][i])
                #print("Neuron value into sigmoid: ", self.neuron_value_holder[layer_index + 1][i])
                self.neuron_normalized_value_holder[layer_index + 1][i] = \
                    self.__sigmoid(self.neuron_value_holder[layer_index + 1][i])
                #print("Neuron Normalized Value after sigmoid: ", self.neuron_normalized_value_holder[layer_index + 1][i])
      
        # just one output neuron, just return one value
        return self.neuron_normalized_value_holder[-1]
        
         
    def caclulate_the_error(self, training_set_outputs):
        # the output layer is the last layer
        # calculate the error for each output neuron
        #print(len(self.neuron_normalized_value_holder))
        
        self.total_error = 0
        
        #print("Output layer: ", self.neuron_normalized_value_holder[-1])
        #print("Expected output: ", training_set_outputs)
            
        out_layer = self.neuron_normalized_value_holder[-1]
        num_of_neruons_in_layer_out = len(self.neuron_normalized_value_holder[-1])
        for i in range(0, num_of_neruons_in_layer_out):
            neuron_error = 0.5 * \
            np.square((training_set_outputs[i] -\
                       self.neuron_normalized_value_holder[-1][i] ))
                
            self.neuron_error_value_holder[i] = neuron_error
            self.total_error += neuron_error
            
        #print("Total error: ", self.total_error)
            
            
    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time. 
    # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    # https://en.wikipedia.org/wiki/Logistic_function#Derivative
    #print("Adjust parameters - input values", linesep,  self.training_set_inputs)
    #print(linesep, "Adjust parameters - weights", linesep,  self.synaptic_weights)              
    def caclulate_backward_pass(self, training_set_outputs):
        # the output layer is the last layer
        # calculate the error for each output neuron
             
        #print("Size of neuron_normalized_value_holder: ", len(self.neuron_normalized_value_holder))
        
        # Walking the layer backwwards
        for layer_index in range(len(self.neuron_normalized_value_holder)-1, 1, -1):
         
            num_neurons_current_layer = len(self.neuron_normalized_value_holder[layer_index])
            num_neurons_in_previous_layer = len(self.neuron_normalized_value_holder[layer_index - 1])
            #print("num_neurons_current_layer: ", num_neurons_current_layer)
            #print("num_neurons_in_previous_layer: ", num_neurons_in_previous_layer)
            
            
            # For each Neuron in the current layer, there is a parameter for the 
            # output of the previosu layer      
            #num_weights_of_parameters_in_neuron = len(self.neuron_weigths_holder[layer_index])
            #print("num_weights_of_parameters_in_neuron: ", num_weights_of_parameters_in_neuron)
            #number_of_parameters_per_neuron = int(num_weights_of_parameters_in_neuron / num_neurons_current_layer)
            #print("number_of_parameters_per_neuron: ", number_of_parameters_per_neuron)
            
            # For each neuro in the current layer
            for i in range(0, num_neurons_current_layer):
                #print("i :", i)
            
                # 1st deal with the neuron's output
                neurons_output = self.neuron_normalized_value_holder[layer_index][i]
                #print("neurons_output: ",neurons_output)
                
            
                # the partial derivative of the total error with respect to 
                # the the neuron's output
                deriv_1 = -1 * (training_set_outputs[i] - neurons_output)
                #print("deriv_1: ", deriv_1)
            
                # the partial derivative of the neuron's output with respect to 
                # the neuron's value
                deriv_2 = neurons_output * (1- neurons_output)
                #print("deriv_2: ", deriv_2)
                
                # num of weights per neuron
                num_weights_of_parameters_in_neuron = len(self.neuron_weigths_holder[layer_index][i])
                
                #print("num_weights_of_parameters_in_neuron: ", num_weights_of_parameters_in_neuron)
                
                #maybe turn this aroudn and let the larger arrays (wights holder )  be on the outside for loop
            
                # For each parameter in a neuron
                for p in range(0, num_weights_of_parameters_in_neuron):
                    #print("p: ", p)
                    # the partial derivative of the neuron's value with respect to
                    # one input parameter
                    # it will be the output of the previous neuron for that parameter
                    deriv_3 = self.neuron_normalized_value_holder[layer_index - 1][i]
                    #print("deriv_3: ", deriv_3)
                    
                    # the derivative of the total error with respect to
                    # the parameter m
                    deriv_10 = deriv_1 * deriv_2 * deriv_3
                    #print("partial derivativ: ", deriv_10 )
                
                    # Adjust the parameter
                    # The parameter belongs to the currrent neuron
#                    self.neuron_weigths_holder[layer_index][i*num_weights_of_parameters_in_neuron + p] -= deriv_10
                
                
                
                
                
            

if __name__ == "__main__":

    # The training set. We have 5 examples, each consisting of 3 input values
    # and 1 expectd output value.
    training_set_inputs = array([[0, 0, 1], 
                                 [0, 1, 0],
                                 [0, 1, 1],
                                 [1, 0, 0]])
    
    training_set_outputs = array([[0,
                                   1,
                                   0,
                                   1]]).T
    
    
    #Intialise a single neuron neural network.
    #number_of_input_neurons, number_of_hidden_neurons, number_of_hidden_layers, number_of_outputs)
    neural_network = NeuralNetwork(3, 4, 2, 1)
    print("Random starting synaptic weights:", linesep)
    print(neural_network.show_synaptic_weights(), linesep)

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training:", linesep)
    print(neural_network.show_synaptic_weights(), linesep)

    # Test the neural network with a new situation.
    test_data = array([[0, 0, 0], 
                       [0, 0, 1],
                       [0, 5, 0],
                       [0, 1, 1],
                       [1, 0, 0],
                       [1, 0, 1],
                       [1, 5, 0],
                       [1, 1, 1]])
    
    for data in test_data:
        print("Considering new situation", data, "-> ",  neural_network.think(data))
    


    
    
    
    