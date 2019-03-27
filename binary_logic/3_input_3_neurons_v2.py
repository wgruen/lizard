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
import os
import sys
import pprint


class NeuralNetwork():
    def __init__(self, number_of_input_neurons, number_of_hidden_neurons, number_of_hidden_layers, number_of_outputs):
        
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(132)
        
        self.training_rate = 0.5
        
        self.neuron_weigths_holder = []
        self.neuron_value_holder = []
        self.neuron_normalized_value_holder = []
        self.neuron_derivative_value_holder = []
        self.output_neuron_error_value_holder = []
        self.neuron_output_partial_product_holder = []
        self.output_neuron_out_minus_target_holder = []


        
        # 1st the input neurons
        weights = np.full(number_of_input_neurons, 999)
        self.neuron_weigths_holder.append(weights) 
        
        value = np.full(number_of_input_neurons, 999) 
        self.neuron_value_holder.append(value)  
        
        neuron_derivative = np.full(number_of_input_neurons, 999) 
        self.neuron_derivative_value_holder.append(neuron_derivative)
         
        input_layer = np.full(number_of_input_neurons, 7.3)
        self.neuron_normalized_value_holder.append(input_layer)
        
        
        # the hidden layer neurons
        number_of_input_neurons = len(input_layer)
        
        # 1st hidden layer uses the inout neurons and the number can be different           
        weights = np.random.rand(number_of_hidden_neurons, number_of_input_neurons)
        self.neuron_weigths_holder.append(weights) 
        
        value = np.full(number_of_hidden_neurons, 8.2) 
        self.neuron_value_holder.append(value)
        
        neuron_derivative = np.zeros(number_of_hidden_neurons) 
        self.neuron_derivative_value_holder.append(neuron_derivative)
        
        normalized_value = np.full(number_of_hidden_neurons, 9.1)
        self.neuron_normalized_value_holder.append(normalized_value)
        
        # more hidden layers have the same number of neurons
        for i in range(0, number_of_hidden_layers -1):
            weights = np.random.rand(number_of_hidden_neurons, number_of_hidden_neurons)
            self.neuron_weigths_holder.append(weights) 
        
            value = np.zeros(number_of_hidden_neurons) 
            self.neuron_value_holder.append(value)
            
            neuron_derivative = np.zeros(number_of_hidden_neurons) 
            self.neuron_derivative_value_holder.append(neuron_derivative)
        
            normalized_value = np.full(number_of_hidden_neurons, 9)
            self.neuron_normalized_value_holder.append(normalized_value)
        
        #The output layer
        weights = np.random.rand(number_of_outputs, number_of_hidden_neurons)
        self.neuron_weigths_holder.append(weights) 
        
        value = np.zeros(number_of_outputs) 
        self.neuron_value_holder.append(value)
        
        normalized_value = np.ones(number_of_outputs) 
        self.neuron_normalized_value_holder.append(normalized_value)
        
        neuron_derivative = np.ones(number_of_outputs) 
        self.neuron_derivative_value_holder.append(neuron_derivative)
      
        error_matrix = np.full(number_of_outputs, 4.6) 
        self.output_neuron_error_value_holder.append(error_matrix)
        
        partial_product = np.full(number_of_outputs, 2.8)
        self.neuron_output_partial_product_holder.append(partial_product)
    
    
        error_derivative = np.full(number_of_outputs, 3.7) 
        self.output_neuron_out_minus_target_holder.append(error_derivative)
        #exit -1
        self.show_matrices()
  
        
        

    def show_matrices(self):
       # return 
        pp = pprint.PrettyPrinter(indent=4)
        print("\n\n")
        print("\nweights holder:")
        pp.pprint(self.neuron_weigths_holder)
        print("\nValue holder: ")
        pp.pprint(self.neuron_value_holder)
        print("\nNormalized value holder:")
        pp.pprint(self.neuron_normalized_value_holder)
 
        print("\nDerivative value holder:")
        print(self.neuron_derivative_value_holder)
        
        print("Output neuron error holder:")
        print(self.output_neuron_error_value_holder)
        print("\nOutput neuron out_minus_target error holder:")
        print(self.output_neuron_out_minus_target_holder)
        print("\n\n")

           
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
            
                # think
                self.think(array)
               
                self.show_matrices()
                     
                # Calculate the error (The difference between the desired output
                # and the predicted output).
                self.caclulate_the_error(training_set_outputs[i])
                
                self.show_matrices()
                
                self.calulate_needed_values_for_backpass(training_set_outputs[i])
                
                self.caclulate_backward_pass_for_ouput_neurons(training_set_outputs[i])
                
                self.caclulate_backward_pass_for_hidden_layers(training_set_outputs[i])
            
                self.show_matrices()
                #break
            #break
            
  
    # Pass inputs through our neural network (our single neuron).
    # use the Dot product
    # https://en.wikipedia.org/wiki/Dot_product 
    def think(self, inputs):
        # each input paramter will have its own neuron
       
        # 1st copy the input values into the Input Neurons
        self.neuron_normalized_value_holder[0] = inputs
        #print("think - Layer normalized value holder, input into the hidden layer:")
        #print(self.neuron_normalized_value_holder[0])
        
        # Now the input neuons are set, let's run the engine
        # Start at the input layer and set the values in the next layer
        # Don't try the last layer, it is the output layer and there is nothing after it
        for current_layer_index in range(0, len(self.neuron_normalized_value_holder)-1):
            
            next_layer_index = current_layer_index + 1
            
            out_layer = self.neuron_normalized_value_holder[current_layer_index]  
            in_layer  = self.neuron_value_holder[next_layer_index]    
            #print("think - Now do layer index %i"  % current_layer_index)
            #print("think - layer OUT - values for the dot product: ", out_layer)

            
            num_of_neruons_in_layer_out = len(out_layer)
            num_of_neruons_in_layer_in = len(in_layer)
            #print("think - Number of neruons in layer OUT: ", num_of_neruons_in_layer_out)
            #print("think - Number of neurons in layer IN: ", num_of_neruons_in_layer_in, linesep, linesep)
            
            # for each neuron
            for i_neuron in range(0, num_of_neruons_in_layer_in):
                # debug        
                #self.show_matrices()
                
                #print("think - Dot product: ",  dot(out_layer, \
                 #            self.neuron_weigths_holder[next_layer_index][i_neuron]))
    
                self.neuron_value_holder[next_layer_index][i_neuron] = \
                         dot(out_layer, \
                             self.neuron_weigths_holder[next_layer_index][i_neuron])
                #print("think - Neuron Value after dot product: ", self.neuron_value_holder[next_layer_index][i_neuron])

                
          
                self.neuron_normalized_value_holder[next_layer_index][i_neuron] = self.__sigmoid(self.neuron_value_holder[next_layer_index][i_neuron])
                #print("think - Neuron Normalized Value after sigmoid: ", self.neuron_normalized_value_holder[next_layer_index][i_neuron])
                
                #break
                
        return self.neuron_normalized_value_holder[-1]
      
         
    def caclulate_the_error(self, training_set_outputs):
        # the output layer is the last layer
        # calculate the error for each output neuron
        
        self.total_error = 0
        
        #print("caclulate_the_error - Output neuron: ", self.neuron_normalized_value_holder[-1])
        #print("caclulate_the_error - Expected output: ", training_set_outputs)
            
        out_layer = self.neuron_normalized_value_holder[-1]
        num_of_neruons_in_layer_out = len(self.neuron_normalized_value_holder[-1])
        for i_neuron in range(0, num_of_neruons_in_layer_out):
            neuron_error = 0.5 * \
            np.square((training_set_outputs[i_neuron] -\
                       self.neuron_normalized_value_holder[-1][i_neuron] ))
                
            self.output_neuron_error_value_holder[i_neuron] = neuron_error
            self.total_error += neuron_error
            
        #print("caclulate_the_error - Total error: ", self.total_error)
            
            
        
    def calulate_needed_values_for_backpass(self, training_set_outputs):
    
        #print("\ncalulate_nvfbp - parameter training_set_outputs : ", training_set_outputs)
        
        # skip the input layer
        # the derivatve of the the neuron, with respect to the neuron's value
        for current_layer_index in range(1, len(self.neuron_normalized_value_holder)):
            print("\ncalulate_nvfbp - current_layer_index: ", current_layer_index)
            
            # for each neuron
            for i_neuron in range(0, len(self.neuron_normalized_value_holder[current_layer_index])):         
                print("calulate_nvfbp - next neuron index: ", i_neuron)
                
                neuron_out = self.neuron_normalized_value_holder[current_layer_index][i_neuron]
                
                self.neuron_derivative_value_holder[current_layer_index][i_neuron] = \
                        neuron_out * ( 1 - neuron_out)
                print("calulate_nvfbp - derivtive of the Neuron's output with respect to the neuron's value: ", self.neuron_derivative_value_holder[current_layer_index][i_neuron])
    
    
        # the derivatve of the Total error  with respect to the neuron's output            
        # for the output neurons only  
        number_of_output_neurons = len(self.neuron_normalized_value_holder[-1])
        print("\n\ncalulate_nvfbp - index_to_output_neurons: ", number_of_output_neurons)
      
        
        for i_neuron in range(0, number_of_output_neurons):
            
            print("calulate_nvfbp - Next output neuron: ", i_neuron)
            neuron_out = self.neuron_normalized_value_holder[-1][i_neuron]
            print("calulate_nvfbp - neuron out: ", neuron_out)
            
            self.output_neuron_out_minus_target_holder[i_neuron] =  \
                neuron_out - training_set_outputs[i_neuron]
                
            print("calulate_nvfbp - derivtive of the Total error with respect to the neuron's output: ", self.neuron_derivative_value_holder[current_layer_index][i_neuron])
            
            self.neuron_output_partial_product_holder[i_neuron] = \
                self.output_neuron_out_minus_target_holder[i_neuron] * self.neuron_derivative_value_holder[-1][i_neuron]
                
    
    
    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time. 
    # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    # https://en.wikipedia.org/wiki/Logistic_function#Derivative
    #print("Adjust parameters - input values", linesep,  self.training_set_inputs)
    #print(linesep, "Adjust parameters - weights", linesep,  self.synaptic_weights)   
    # https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c           
    def caclulate_backward_pass_for_ouput_neurons(self, training_set_outputs):
        # the output layer is the last layer
        # calculate the error for each output neuron
        # Walking the layer backwwards
        
   
        index_to_output_neurons = len(self.neuron_normalized_value_holder)
        #print("Backpass output neurons - index_to_output_neurons: ", index_to_output_neurons)
        
        for current_layer_index in range(index_to_output_neurons-1, index_to_output_neurons-2, -1):
         
            num_neurons_current_layer = len(self.neuron_normalized_value_holder[current_layer_index])
            num_neurons_in_previous_layer = len(self.neuron_normalized_value_holder[current_layer_index - 1])
            
            #print("Backpass output neurons - num_neurons_current_layer: ", num_neurons_current_layer)
            #print("Backpass output neurons - num_neurons_in_previous_layer: ", num_neurons_in_previous_layer)
            
            
            # For each Neuron in the current layer, there is a parameter for the 
            # output of the previosu layer      
            #num_weights_of_parameters_in_neuron = len(self.neuron_weigths_holder[current_layer_index])
            #print("num_weights_of_parameters_in_neuron: ", num_weights_of_parameters_in_neuron)
            #number_of_parameters_per_neuron = int(num_weights_of_parameters_in_neuron / num_neurons_current_layer)
            #print("number_of_parameters_per_neuron: ", number_of_parameters_per_neuron)
            
            # For each neuro in the current layer
            for i_neuron in range(0, num_neurons_current_layer):
                print("\n\nBackpass output neurons - neuron :", i_neuron)
        
                         
                # the partial derivative of the total error with respect to 
                # the the Ouput Neuron's output
                deriv_1 = self.output_neuron_out_minus_target_holder[i_neuron]
                print("Backpass output neurons - deriv_1: ", deriv_1)
            
                # the partial derivative of the neuron's output with respect to 
                # the neuron's value
                deriv_2 = self.neuron_derivative_value_holder[current_layer_index][i_neuron]
                print("Backpass output neurons - deriv_2: ", deriv_2)
                
                # num of weights per neuron
                num_weights_in_neuron = len(self.neuron_weigths_holder[current_layer_index][i_neuron])
                print("Backpass output neurons - num_weights_of_parameters_in_neuron: ", num_weights_in_neuron)
                
                # For each parameter in a neuron
                for p_index in range(0, num_weights_in_neuron):
                    print("\nBackpass output neurons - parameter: ", p_index)
                    
                    # the partial derivative of the neuron's value with respect to
                    # one input parameter
                    # it will be the output of the previous neuron for that parameter
                    deriv_3 = self.neuron_normalized_value_holder[current_layer_index - 1][p_index]
                    print("Backpass output neurons - deriv_3: ", deriv_3)
                    
                    # the partial derivative of the total error with respect to
                    # the parameter m
                    deriv_10 = deriv_1 * deriv_2 * deriv_3
                    print("Backpass output neurons - partial derivativ 10: ", deriv_10 )
                
                    # Adjust the parameter
                    # The parameter belongs to the currrent neuron
                    print("Backpass output layer - weight before adjustment: ", self.neuron_weigths_holder[current_layer_index][i_neuron][p_index])       
                    current_weight = self.neuron_weigths_holder[current_layer_index][i_neuron][p_index]                                      
                    self.neuron_weigths_holder[current_layer_index][i_neuron][p_index] = current_weight - self.training_rate * deriv_10
                    print("Backpass output layer - weight after adjustment: ", self.neuron_weigths_holder[current_layer_index][i_neuron][p_index])
                    
                    
    ''' this works for one hidden layer only '''            
    def caclulate_backward_pass_for_hidden_layers(self, training_set_outputs):
        # the output layer is the last layer
        # calculate the error for each output neuron
        # Walking the layer backwwards

        index_to_output_neurons = len(self.neuron_normalized_value_holder)
        print("\n\nBackpass hidden layers - index_to_output_neurons: ", index_to_output_neurons)
        
        # hidden layers start at layer -2 and do not include the first layer zero
        for current_layer_index in range(1, 0, -1):
            
            # skip the input layer]
            if current_layer_index is 0:
                break
         
            num_neurons_current_layer = len(self.neuron_normalized_value_holder[current_layer_index])
            num_neurons_in_previous_layer = len(self.neuron_normalized_value_holder[current_layer_index - 1])
            
            print("Backpass hidden layers - num_neurons_current_layer: ", num_neurons_current_layer)
            print("Backpass hidden layers - num_neurons_in_previous_layer: ", num_neurons_in_previous_layer)
            
            #self.show_matrices()
    
            # For each neuro in the current layer
            for i_neuron in range(0, num_neurons_current_layer):
                print("\n\nBackpass hidden layers - neuron :", i_neuron)
            
                # the derivative of the Total Error with respect to the neruon's output              
                number_of_output_neurons = len(self.neuron_normalized_value_holder[-1])
                print("Backpass hidden layers  - number_of_output_neurons: ", number_of_output_neurons)
    
                deriv_total_error_wrt_neurons_output = 0
                for i_output in range(0, number_of_output_neurons):
                    
                    # the derivative of the Output Neruon's value with respect to the previous neuron's output
                    # is the weight for this specfic neuron
                    ouput_neuron_weight = self.neuron_weigths_holder[-1][i_output][i_neuron]
                    print("Backpass hidden layers  - ouput_neuron_weight: ", ouput_neuron_weight) 
                    
                    
                    # the partial derivative of te next Neurons oupput with respect to the neurons value
                    deriv11 =  self.neuron_output_partial_product_holder[i_output] * ouput_neuron_weight
                    deriv_total_error_wrt_neurons_output += deriv11
                
                print("Backpass hidden layers  - deriv_total_error_wrt_neurons_output: ", deriv_total_error_wrt_neurons_output)   
                
                # num of weights per neuron
                num_weights_in_neuron = len(self.neuron_weigths_holder[current_layer_index][i_neuron])              
                print("Backpass hidden layers  - num_weights_of_parameters_in_neuron: ", num_weights_in_neuron)
            
                # For each parameter in a neuron
                for i_parameter in range(0, num_weights_in_neuron):
                    print("Backpass hidden layers - neuron %d   parameter %d " %(i_neuron, i_parameter))
        
        
                    # the partial derivative of the neuron's output with respect to
                    # neuron's value
                    deriv_1 = self.neuron_derivative_value_holder[current_layer_index][i_neuron]
                    print("Backpass hidden layers - deriv_1: ", deriv_1)
                    
                    # the partial derivative of the neuron's value with respect to
                    # one input parameter
                    # it will be the output of the previous neuron for that parameter
                    deriv_2 = self.neuron_normalized_value_holder[current_layer_index - 1][i_parameter]
                    print("Backpass hidden layers - deriv_2: ", deriv_2)
                    
                    #the derivative of the neruo
                    
                    # the partial derivative of the total error with respect to
                    # the parameter m
                    deriv_10 = deriv_1 * deriv_2 * deriv_total_error_wrt_neurons_output
                    print("Backpass hidden layers - partial derivativ 10: ", deriv_10 )
                
                    # Adjust the parameter
                    # The parameter belongs to the currrent neuron
                    print("Backpass hidden layers - Backpass weight before adjustment: ", self.neuron_weigths_holder[current_layer_index][i_neuron][i_parameter])
                    current_weight = self.neuron_weigths_holder[current_layer_index][i_neuron][i_parameter]
                    
                    self.neuron_weigths_holder[current_layer_index][i_neuron][i_parameter] = current_weight - self.training_rate * deriv_10
                    print("Backpass hidden layers - Backpass weight after adjustment: ", self.neuron_weigths_holder[current_layer_index][i_neuron][i_parameter])
                    

                
            

if __name__ == "__main__":

    # The training set. We have 5 examples, each consisting of 3 input values
    # and 1 expectd output value.
    training_set_inputs = array([[0, 0, 1], 
                                 [1, 1, 0],
                                 [1, 0, 0],
                                 [0, 1, 1]])
    
    training_set_outputs = array([[0,
                                   1,
                                   0,
                                   1]]).T
    
    
    #Intialise a single neuron neural network.
    #number_of_input_neurons, number_of_hidden_neurons, number_of_hidden_layers, number_of_outputs)
    neural_network = NeuralNetwork(3, 3, 1, 1)
    
    
  
    print("Random starting synaptic weights:", linesep)
    print(neural_network.show_synaptic_weights(), linesep)
    
    
    #sys.stdout = open(os.devnull, 'w')

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10000)
    
    #sys.stdout = sys.__stdout__
    

    print("New synaptic weights after training:", linesep)
    print(neural_network.show_synaptic_weights(), linesep)
    print("All matrixes: ", neural_network.show_matrices())

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
    


    
    
    
    