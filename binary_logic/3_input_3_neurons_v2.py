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
    def __init__(self, number_of_input_neurons, number_of_hidden_neurons, number_of_outputs):
        
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)
        
        self.input_neurons = []
        self.hidden_neurons = []
        self.output_neurons = []
        
        # the inut neurons
        self.neuron_input_holder = []
        self.neuron_weigths_holder = []
        self.neuron_value_holder = []
        self.neuron_normalized_value_holder = []

        
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
        #print(input_layer)
        #exit -1
        
        # the hidden layer neurons
        size_input_layer = len(input_layer)
        
        layer = np.zeros(number_of_hidden_neurons * size_input_layer)
        self.neuron_input_holder.append(layer) 
               
        weights = np.random.rand(number_of_hidden_neurons * size_input_layer)
        self.neuron_weigths_holder.append(weights) 
        
        value = np.zeros(number_of_hidden_neurons) 
        self.neuron_value_holder.append(value)
        
        normalized_value = np.full(number_of_hidden_neurons, 9)
        self.neuron_normalized_value_holder.append(normalized_value)
        
        #The output layer
       # the hidden layer neurons
        layer = np.zeros(number_of_outputs * number_of_hidden_neurons)
        self.neuron_input_holder.append(layer) 
               
        weights = np.random.rand(number_of_outputs * number_of_hidden_neurons)
        self.neuron_weigths_holder.append(weights) 
        
        value = np.zeros(number_of_outputs) 
        self.neuron_value_holder.append(value)
        
        normalized_value = np.ones(number_of_outputs) 
        self.neuron_normalized_value_holder.append(normalized_value)
        
    
        print("Input holder: ", self.neuron_input_holder)
        print("Weights holder: ", self.neuron_weigths_holder)
        print("Value holder:  ", self.neuron_value_holder)
        print("Normalized value holder: ", self.neuron_normalized_value_holder)
        #exit -1
  
        
        
        # OLD below
        # only one input for the input neurons
        for i in range(number_of_input_neurons):
            self.input_neurons.append(Neuron(0, 1))
            
            
        for i in range(number_of_input_neurons):
            self.hidden_neurons.append(Neuron(1, number_of_input_neurons)) 
            
        # only one output neuron
        for i in range(1):
            self.output_neurons.append(Neuron(2, number_of_input_neurons)) 

        print("Done creating all neurons")
        #print(self.input_neurons)
        #print(self.hidden_neurons)
        #print(self.output_neurons)
  

    def show_matrices(self):
        print("Input holder: ", self.neuron_input_holder)
        print("Weights holder: ", self.neuron_weigths_holder)
        print("Value holder:  ", self.neuron_value_holder)
        print("Normalized value holder: ", self.neuron_normalized_value_holder)

          
           
    def show_synaptic_weights(self):
        return 
    
        print("Input Neuron weights:")
        for neuron in self.input_neurons:
            neuron.print_synaptic_weights()
        
        print("Hidden Neuron weights:")
        for neuron in self.hidden_neurons:
            neuron.print_synaptic_weights()
            
        print("Output Neuron weights:")  
        for neuron in self.output_neurons:
            neuron.print_synaptic_weights()
 
    
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
                print("Neural Network output: ",  nn_output)
                #continue##wolfwolf
                     
                # Calculate the error (The difference between the desired output
                # and the predicted output).
                error = expected_output - nn_output
                error_total = 0.5 * pow(error, 2)
                
                # this specific machine uses one ouput neuron 
                # and therefore the backward pass is simpler
    
                
                # 1st we update the last layer, the output layer
                need_for_next_layer = 0
                for i6 in range(len(self.output_neurons)):
                    error = self.output_neurons[i6].train_adjust_synaptic_weights_for_output_neuron(expected_output)
                    need_for_next_layer = self.output_neurons[i6].derived_normalized_vs_value * \
                        self.output_neurons[i6].out_minus_target
                    
                # Next the hidden layer
                
                for i7 in range(len(self.hidden_neurons)):
                    error = self.hidden_neurons[i7].train_adjust_synaptic_weights_for_hidden_neuron(need_for_next_layer)
                           
                # Last the input layer
                #for i8 in range(0, len(self.input_neurons)):
                 #   error = self.input_neurons[i8].train_adjust_synaptic_weights(out_minus_target)              
             
                #break
            #break
            
  #
        
    def think(self, inputs):
        # each input paramter will have its own neuron
        number_of_inputs = len(inputs)
   
        print("Think Input value: ", inputs)
            
    
        
        # 1st copy the input values into the Input Neurons
        print(self.neuron_normalized_value_holder[0])
        self.neuron_normalized_value_holder[0] = inputs
        print(self.neuron_normalized_value_holder[0])
        
        # run the engine
 
        print(len(self.neuron_normalized_value_holder))
        for layer_index in range(0, len(self.neuron_normalized_value_holder)-1):
            
            #copy the output from the previous layer, as input for the next layer
            out_layer = self.neuron_normalized_value_holder[layer_index]
            in_layer = self.neuron_input_holder[layer_index + 1]
            
            print("layer values out: ", out_layer)
            print("layer values in: ", in_layer)
            #break
            
            out_layer_size = len(self.neuron_normalized_value_holder[layer_index])
            in_layer_size = len(self.neuron_normalized_value_holder[layer_index + 1])
            
            print("out layer size: ", out_layer_size)
            print("in layer size: ", in_layer_size, linesep, linesep)
            
            for i in range(0, in_layer_size):
                # copy the normalized output value from the previous layer into
                # the input values for the next layer
                
                
                self.neuron_input_holder[layer_index + 1][i*out_layer_size: (i*out_layer_size) + out_layer_size] = out_layer
                #in_layer[0,0 :0:2] = out_layer
                print(1 , " after in layer: ", self.neuron_input_holder[layer_index + 1])
                
                # debug        
                self.show_matrices()
                
                # Pass inputs through our neural network (our single neuron).
                # use the Dot product
                # https://en.wikipedia.org/wiki/Dot_product
                self.neuron_value_holder[layer_index + 1][i] = \
                         dot(self.neuron_input_holder[layer_index + 1][i*out_layer_size: (i*out_layer_size) + out_layer_size], \
                           self.neuron_weigths_holder[layer_index + 1][i*out_layer_size: (i*out_layer_size) + out_layer_size])
                
                
                
                
                self.neuron_normalized_value_holder[layer_index + 1][i] = \
                    self.__sigmoid(self.neuron_value_holder[layer_index + 1][i])
                
                
        
        #self.normalized_neuron_value = self.__sigmoid(self.internal_neuron_value)
        #print("After sigmoid:", linesep, self.normalized_neuron_value, linesep)
              
        
        
        # just one output neuron, just return one value
        return self.neuron_normalized_value_holder[-1]
        
                    

class Neuron():
    # give each Neuron a unique number
    static_neuron_number = 1
    
    #layer 0 is the inut layer
    def __init__(self, neuron_layer_number, number_of_inputs):
        self.neuron_layer_number = neuron_layer_number
        self.neuron_number = Neuron.static_neuron_number
        Neuron.static_neuron_number += 1
        

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = []
        if number_of_inputs is 1:
            self.synaptic_weights.append([1.0])
        else:
            self.synaptic_weights = 2 * random.random((number_of_inputs, 1)) - 1
            
        self.internal_neuron_value = 0
        self.normalized_neuron_value = 0
        self.training_rate = 0.2
        self.out_minus_target = 0
        self.derived_normalized_vs_value = 0


    def __repr__(self):
        return "The neuron layer #: " + str(self.neuron_layer_number) + \
        "\tneuron # :"  + str(self.neuron_number) + linesep
   
    #def __str__(self):
      #  return "The neuron layer #: " + str(self.neuron_layer_number) + "\tneuron # :"  + str(self.neuron_number) + linesep
    
    
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


    def print_synaptic_weights(self):
        print("Neuron layer #: " + str(self.neuron_layer_number) + "\tneuron # :"  + \
              str(self.neuron_number) + "\t weight" + str(self.synaptic_weights) + \
              "\t output: " + str(self.normalized_neuron_value))
        

    # Pass inputs through our neural network (our single neuron).
    # use the Dot product
    # https://en.wikipedia.org/wiki/Dot_product
    def think_brain(self, in_value):
        self.training_set_inputs = in_value
        
        print("Input: ", in_value, linesep)
        print("Weights: ", self.synaptic_weights, linesep)
                
                
        # Pass the training set through our neural network (a single neuron).
        self.internal_neuron_value = dot(in_value, self.synaptic_weights)
        #print("Dot product:", linesep, self.internal_neuron_value, linesep)
        
        self.normalized_neuron_value = self.__sigmoid(self.internal_neuron_value)
        #print("After sigmoid:", linesep, self.normalized_neuron_value, linesep)
       
        return self.normalized_neuron_value


    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train_adjust_synaptic_weights_for_output_neuron(self, output_neuron_value, ):
        # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        # https://en.wikipedia.org/wiki/Logistic_function#Derivative
        #print("Adjust parameters - input values", linesep,  self.training_set_inputs)
        #print(linesep, "Adjust parameters - weights", linesep,  self.synaptic_weights)
        
        
        self.out_minus_target = -1 * (output_neuron_value - self.normalized_neuron_value)
        self.derived_normalized_vs_value = self.normalized_neuron_value * ( 1 - self.normalized_neuron_value)
        
        for i in range(0, len(self.synaptic_weights)):
            weight = self.synaptic_weights[i]
            derived_value_vs_parameter = self.training_set_inputs[i]
            partial_derived_for_parameter = \
                self.out_minus_target * \
                self.derived_normalized_vs_value * \
                derived_value_vs_parameter
                
            
            
            self.synaptic_weights[i] = weight - self.training_rate * partial_derived_for_parameter
        #print("Adjust parameters - input values", linesep,  self.training_set_inputs)         
        #print("Adjust parameters - new weights", linesep,  self.synaptic_weights) 
    

   # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train_adjust_synaptic_weights_for_hidden_neuron(self, needed_value):
        # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        # https://en.wikipedia.org/wiki/Logistic_function#Derivative
        #print("Adjust parameters - input values", linesep,  self.training_set_inputs)
        #print(linesep, "Adjust parameters - weights", linesep,  self.synaptic_weights)
        
        self.out_minus_target = needed_value
        self.derived_normalized_vs_value = self.normalized_neuron_value * ( 1 - self.normalized_neuron_value)
        
        for i in range(0, len(self.synaptic_weights)):
            weight = self.synaptic_weights[i]
            derived_value_vs_parameter = self.training_set_inputs[i]
            partial_derived_for_parameter = \
                self.out_minus_target * weight * \
                self.derived_normalized_vs_value * \
                derived_value_vs_parameter
                
            
            
            self.synaptic_weights[i] = weight - self.training_rate * partial_derived_for_parameter
        #print("Adjust parameters - input values", linesep,  self.training_set_inputs)         
        #print("Adjust parameters - new weights", linesep,  self.synaptic_weights) 
    

    # The neural network thinks.
    def think(self, inputs):
        thinking_results = self.think_brain(inputs)
        return thinking_results, round(thinking_results)
        

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
    neural_network = NeuralNetwork(3, 3, 1)
    print("Random starting synaptic weights:", linesep)
    print(neural_network.show_synaptic_weights(), linesep)

    # Train the neural network using a training set.
    # Do it 10,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10)

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
    
    #for data in test_data:
        #print("Considering new situation", data, "-> ",  neural_network.think(data))
    


    
    
    
    