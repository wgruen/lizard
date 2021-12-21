#!/usr/bin/env python3


import sys
from matplotlib import pyplot
import argparse
import yaml
import json
import pprint
import io
import os
from os import linesep
from contextlib import redirect_stdout
import random
import numpy as np
import decimal
import string


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from reportlab.pdfgen import canvas
from reportlab.platypus import *
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
styles = getSampleStyleSheet()


# example of loading the cifar10 dataset
import tensorflow as tf
from tensorflow import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from keras.regularizers import l2
import keras.optimizers 


class binary_with_keras():
    def __init__(self, configuration):
        ### get the parameters
        self.configuration = configuration
        self.verbose = configuration["machine"]["verbose"]
        self.dropout_percentage = configuration["dropout_percentage"] / 100
        self.mkernel_regularizer_l2 = configuration["mkernel_regularizer_l2"]
        self.number_of_epochs = configuration["machine"]["number_of_epochs"]
        self.machine_dump_file_name_base = configuration["machine"]["machine_dump_file_name_base"]
        self.set_file_names()
        

    def set_file_names(self):
        number_of_input_neurons   = self.configuration["machine"]["number_of_input_neurons"]
        number_of_hidden_neurons_layers_1  = self.configuration["machine"]["number_of_hidden_neurons_layers_1"]
        number_of_hidden_neurons_layers_2   = self.configuration["machine"]["number_of_hidden_neurons_layers_2"]

        
        self.model_file_name_base = "".join(( self.machine_dump_file_name_base,
            "--ep-",
            str(self.number_of_epochs), 
            "--i-",
            str(number_of_input_neurons),
            "--hl1-",
            str(number_of_hidden_neurons_layers_1),
            "--hl2-",
            str(number_of_hidden_neurons_layers_2)
            ))
             
             
 
        self.model_file_name = self.model_file_name_base + ".bin"
        self.pdf_plots_file_name  = self.model_file_name_base + "_plots.pdf"
        self.doc_summary_file_name  = self.model_file_name_base  + "_summary.pdf"
        
            
    '''
    load the dataset
    '''
    def load_dataset(self):
        # Load the training data
        training_set_inputs = []
        
        if "training_data_file_name" in self.configuration["training_data"]:
            # load from file
            file_name = self.configuration["training_data"]["training_data_file_name"]
            print("Input file name: "+ file_name)
            training_set_inputs = np.loadtxt(file_name)
        else:
            training_data_set = self.configuration["training_data"]["train_with_embedded"]         
            training_set_inputs = self.configuration["training_data"][training_data_set]
            training_set_inputs = np.array(training_set_inputs)
            
        
        
        print("Raw training_set_inputs: ****************" + linesep + str(training_set_inputs))
        # extract input data and expected output
        self.trainY = training_set_inputs[:,-1:] # for last column
        self.trainX =  training_set_inputs[:, :-1] # for all but last column
        # Tur output into proper format
        #        training_set_outputs = training_set_outputs.reshape(training_set_outputs.shape[0], 1)
        
        print("training_set_inputs: ****************"  + linesep + str(self.trainX))
        print("training_set_outputs: ****************" + linesep + str(self.trainY))
        
    
        # load the validation  data
        if "validation_data_file_name" in self.configuration:
            test_data = np.loadtxt(self.configuration["validation_data_file_name"])
        else:
            test_data = self.configuration["validation_data_embedded"]
            test_data = np.array(test_data)

            
        #test_data = np.array(test_data, dtype=np.dtype(decimal.Decimal))
        self.testY = test_data[:,-1:] # for last column
        self.testX = test_data[:, :-1] # for all but last column
    
        # convert into form that can be used to ML algorithm
        # one hot encode target values
        # to_categorical returns a binary matrix
#        self.trainY = to_categorical(self.trainY)
      #  self.trainX = self.trainX.reshape(-1, 1)
       # self.trainX = self.trainX.reshape(3, -1)

#        self.testY = to_categorical(self.testY)    


    def print_dataset(self):
        # summarize loaded dataset
        
        print("self.trainX")
        print(self.trainX)

        print("self.trainY")
        print(self.trainY)
        
        print("self.testX")
        print(self.testX)
        
        print("self.testY")
        print(self.testY)
    
    
    '''
    The integers are in the range of 0 to 255
    Here we normalize the picture values.
    Rescale them to the range [0,1]
    
    This prepares the pixel data
    '''
    def prepare_data(self, data):
        # convert from integers to floats
#        data_norm = data.astype('float32')
        
        # normalize to range 0-1
 #       data_norm = data_norm / 255.0
        
        # return normalized images
#        return data_norm
        return data
 
       
    '''
    Define the keras model.
    Please note that the model defintion is highly dependenet on the 
    image size
    
    Here we build the layers of the CNN (Convolutional Neural Networks) model
    CCN was designed to work with two dimensional image data.
    '''
    def define_model(self):
        number_of_input_neurons   = self.configuration["machine"]["number_of_input_neurons"]
        number_of_hidden_neurons_layers_1  = self.configuration["machine"]["number_of_hidden_neurons_layers_1"]
        number_of_hidden_neurons_layers_2   = self.configuration["machine"]["number_of_hidden_neurons_layers_2"]
        number_of_outputs_neurons = self.configuration["machine"]["number_of_outputs_neurons"]
        my_learning_rate = self.configuration["machine"]["learning_rate"]
        my_loss = self.configuration["machine"]["loss_function"] 
        my_bias_initializer = self.configuration["machine"]["bias_initializer"] 
        my_metrics = self.configuration["machine"]["metrics"] 
        my_optimizer = self.configuration["machine"]["optimizer"] 
       

        
        # https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
        
        self.model = Sequential()
       
        # input shape
        # There is one data point for each input
        # and there should be three Neurons
        #
        # the first hidden layer has 4 nodes
        
        self.model.add(Dense(
            number_of_hidden_neurons_layers_1,
            input_dim=number_of_input_neurons,
            activation='relu',
            bias_initializer=my_bias_initializer))
        
        # One hidden layer should be sufficient for the 
        # majority of problems :-)
        # Hidden layers are required only, and only if the data 
        # must be separated non-linearly
        if number_of_hidden_neurons_layers_2 is 1:
            self.model.add(Dense(
                number_of_hidden_neurons_layers_2, 
                activation='relu',
                bias_initializer=my_bias_initializer))
        
        # The output layer should be one Neuron
        self.model.add(Dense(number_of_outputs_neurons, activation='sigmoid'))

        #the_optimizer = tf.compat.v1.train.AdamOptimizer(0.01) # just set a default
        #tf.keras.optimizers.Adam(0.01) 
        
        # pick an optimizer for gradient descent with momentum optimizer
        print("my_optimizer")
        print(my_optimizer)
    
        if my_optimizer is 'SGD':
            self.the_optimizer = SGD(lr=0.001, momentum=0.9)
        
        if my_optimizer == 'adam':
            print("optimizer adams was choosen")
            self.opt = tf.keras.optimizers.Adam(learning_rate=my_learning_rate)
        
        #self.opt = tf.keras.optimizers.Adam(learning_rate=my_learning_rate)
        
        # compile the model
        #https://keras.io/api/metrics/
        self.compile_output = self.model.compile(run_eagerly=True, optimizer=self.opt, loss=my_loss,  metrics=[my_metrics]) 
        
        print(self.model.summary()) 
     #   sys.exit()
     
    
    '''
    The model will be trained (fit) and
    the trained model will be stored to hd.
    
    '''
    def fit_model(self):
        # set the file names for a run
     #   self.set_file_names()
        
        ### load and prepare the dataset
        self.load_dataset()
        self.print_dataset()
        
        # normalize the pixel data for X only
        self.trainX = self.prepare_data(self.trainX)
        self.testX = self.prepare_data(self.testX)
        
        ### Define and use a model
        self.define_model()
 

        # todo print to pdf before fitting
        
        ### FIT / TRAIN the model
        ## Sample - is a set of data, also called Rows Of Data
        # In the case of CIFAR-10, as sample is the same as the data of one picture
        # , I guess
        #
        ## batch size - is a number of samples (Rows of Data)
        # which are processed before the model is updated.
        # The size of a batch must be more than or equal to one and 
        # less than or equal to the number of samples in the training dataset.
        #
        ## epoch - number of complete passes through the training set
        #
        ## vaidation data
        #  Data on which to evaluate the loss of the model 
        # at the end of each epoch. The model will not be trained on this data.
        # https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
        
        self.print_dataset()
        
        
        validate_data_during_fitting   = self.configuration["machine"]["validate_data_during_fitting"] 
        shuffle_data = self.configuration["machine"]["shuffle_data"]
        batch_size = self.configuration["machine"]["batch_size"]

        
        
        if validate_data_during_fitting is 0:        
            self.fit_history = self.model.fit(self.trainX, self.trainY,\
            epochs=self.number_of_epochs,\
            batch_size=batch_size,\
            #validation_data=(self.testX, self.testY),\
            verbose=self.verbose,
            shuffle=shuffle_data)
        else:
            self.fit_history = self.model.fit(self.trainX, self.trainY,\
            epochs=self.number_of_epochs,\
            batch_size=batch_size,\
            validation_data=(self.testX, self.testY),\
            verbose=self.verbose,
            shuffle=shuffle_data)
    

        # The history dictionary will have these keys: 
        # ['val_loss', 'val_acc', 'loss', 'acc']
            

        # SAVE the trained model
        if not os.path.exists("saved_machines"):
            os.mkdir("saved_machines")
        model_file_name_and_path = os.path.join("saved_machines", self.model_file_name)
        print("path to save model")
        print(model_file_name_and_path)

        tf.keras.models.save_model(self.model, model_file_name_and_path)
        #tf.saved_model.save(self.model, model_file_name_and_path)
    
        self.create_output_pdf()
    
        #self.predict_data()
        
            
    def create_output_pdf(self):
        ##############################################
        ### print plots
        ##############################################
        if not os.path.exists("output"):
            os.mkdir("output")
                     
        pdf_plots  = PdfPages("output/" +  self.pdf_plots_file_name)
        
        # SAVE documentation of the trained model
        # The history.history dictionary will have these keys: 
        # ['val_loss', 'val_acc', 'loss', 'acc']
        # plot loss
        print("fit_history", self.fit_history.history)
        print(self.fit_history.history.keys())
        
    
        
        pyplot.subplot(3, 1, 1)
        pyplot.title('Cross Entropy Loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        pyplot.plot(self.fit_history.history['loss'], color='blue', label='train model')
        if "val_loss" in self.fit_history.history:
            pyplot.plot(self.fit_history.history['val_loss'], color='orange', label='validate model')
        
        
        # plot accuracy
        #pyplot.subplot(3, 1, 3)
        #pyplot.title('Classification Accuracy')
        #plt.ylabel('Accuracy')
        #plt.xlabel('epoch')
        #plt.legend(['train', 'test'], loc='upper left')
        #pyplot.plot(self.fit_history.history['accuracy'], color='blue', label='train_model')
        #pyplot.plot(self.fit_history.history['val_accuracy'], color='orange', label='validate_model')
        
        
        
        # save plot to file
        pyplot.savefig(pdf_plots, format='pdf') #, bbox_inches='tight')
        pyplot.plot()
        pyplot.close()
        pdf_plots.close()
        
        
        ##############################################
        ### create a text document with data
        ##############################################
        doc_summary  = SimpleDocTemplate("output/" + self.doc_summary_file_name, pagesize=letter)
    
        element = []
        header = Paragraph("\nSummary of Training Runs", styles["Heading1"])
        element.append(header)
        
        header = Paragraph("\nThe script's input parameters", styles["Heading2"])
        element.append(header)
        text = yaml.dump(self.configuration, indent=4)
        print(text)
        para = XPreformatted(text, styles["Code"], dedent=0)
        element.append(para)

        header = Paragraph("\nThe class / runs parameters", styles["Heading2"])
        element.append(header)
        text = "dropout_percentage: " + str(self.dropout_percentage) + os.linesep +\
            "mkernel_regularizer_l2: " + str(self.mkernel_regularizer_l2) + os.linesep +\
            "number_of_epochs: " + str(self.number_of_epochs) 
        print(text)
        para = XPreformatted(text, styles["Code"], dedent=0)
        element.append(para)


                
        header = Paragraph("\nThe model summary", styles["Heading2"])
        element.append(header)
        f = io.StringIO() 
        with redirect_stdout(f):
            self.model.summary() 
        s = f.getvalue()
        #print("model summary:\n", s)
        para = XPreformatted(s, styles["Code"], dedent=0)
        element.append(para)  
        

        header = Paragraph("\nThe model metric", styles["Heading2"])
        element.append(header)
        f = io.StringIO() 
        with redirect_stdout(f):
            self.model.metrics_names
        s = f.getvalue()
        #print("model summary:\n", s)
        para = XPreformatted(s, styles["Code"], dedent=0)
        element.append(para)          
        
       
        header = Paragraph("\nTrain X values", styles["Heading2"])
        element.append(header)
        text = str(self.trainX)       
        para = XPreformatted(text, styles["Code"], dedent=0)
        element.append(para)        
        
        header = Paragraph("\nTrain Y values", styles["Heading2"])
        element.append(header)
        text = str(self.trainY)   
        para = XPreformatted(text, styles["Code"], dedent=0)
        element.append(para)   
        
        validate_data_during_fitting = self.configuration["machine"]["validate_data_during_fitting"]
        if validate_data_during_fitting is 1:
            header = Paragraph("\nValidate X values", styles["Heading2"])
            element.append(header)
            text = str(self.trainX)       
            para = XPreformatted(text, styles["Code"], dedent=0)
            element.append(para)   
        
            header = Paragraph("\nValdiate Y values", styles["Heading2"])
            element.append(header)
            text = str(self.trainY)
            para = XPreformatted(text, styles["Code"], dedent=0)
            element.append(para)   
            

        header = Paragraph("\nThe model configuration", styles["Heading2"])
        element.append(header)
        json_formatted_str = json.dumps(self.model.get_config(), indent=2, sort_keys=True)
        #print(json_formatted_str)
        para = XPreformatted(json_formatted_str,  styles["Code"], dedent=0)
        element.append(para)
        
        header = Paragraph("\nThe model layers and weights", styles["Heading2"])
        element.append(header)
        para = XPreformatted(str(self.model.get_weights()),  styles["Code"], dedent=0)
        element.append(para)
            
    
        doc_summary.build(element)


    '''
    The model is evaluated with the test data, which is part of
    the training data
    '''
    def evaluate_data(self):
        ### load and prepare the dataset
        self.load_dataset()
        self.print_dataset()
        
        ## load a previously stored model
        self.model = keras.models.load_model(self.model_file_name)
    
        ## EVALUATE the previously trained model
        self.testX = self.prepare_data(self.testX)
        self.eval = self.model.evaluate(self.testX, self.testY, verbose=self.verbose)
        
        print("=== after eval ===")
        print(self.testX)
        print(self.testY)
        print(self.model.metrics_names)
        print("result below")
        print(self.eval)
        
    '''
    The model is evaluated with the test data, which is part of
    the training data
    '''
    def predict_data(self):
        ### load and prepare the dataset
        print("enter predcit data")
        self.load_dataset()
        self.print_dataset()
        
        ## load a previously stored model
        model_file_name_and_path = os.path.join("saved_machines", self.model_file_name)
        print(model_file_name_and_path)
     
     #   self.my_model_predict = tf.saved_model.load(str(model_file_name_and_path))

        self.my_model_predict = tf.keras.models.load_model(model_file_name_and_path)

        #self.my_model_predict = tf.compat.v1.saved_model.load(model_file_name_and_path)
    
        ## PREDICT with the previously trained model
        self.testX = self.prepare_data(self.testX)
        pred = self.my_model_predict.predict(self.testX, verbose=self.verbose)
      
        print("=== after prediction ===")
        #pred = np.array(pred))
        np.set_printoptions(formatter={'float_kind':"{:.4f}".format})
        pred = pred.astype(np.float64)
        print(pred)
        
        # difference to expectations
        delta = pred - self.testY
        print("the difference to the expected value")
        print(delta)
        
        # show them side by side
        side_by_side = np.dstack((pred, delta))
        print(side_by_side) 
        
        # get the total error
        total_error = np.absolute(delta)
        print("total error")
        print(total_error)
        total_error = sum(total_error)
        print(total_error)

        
        return total_error
        


'''
main
'''
def main(argv):
    parser = argparse.ArgumentParser(description='Training and running a small neural network.')
    parser.add_argument('-i', dest='input_config_file', 
                        action='store', 
                        required=True,
                        help='The yaml file containing configuration')
    args = parser.parse_args()
    #print(args)
       
    configuration = None
    with open(args.input_config_file, 'r') as stream:
        configuration = yaml.safe_load(stream)
        
    print(yaml.safe_dump(configuration, default_flow_style=False, default_style=None))
    
    if(configuration["fit_the_model"] == 1):
        for kernel_regulror_l1l2 in configuration["mkernel_regularizer_l2_range"]:
            configuration["mkernel_regularizer_l2"] = kernel_regulror_l1l2
            for dropout in configuration["dropout_percentage_range"]:
                configuration["dropout_percentage"] = dropout
                mymodel = binary_with_keras(configuration)
                mymodel.fit_model()
            
    if(configuration["predict_the_model"] == 1):
        mymodel = binary_with_keras(configuration)
        mymodel.predict_data()

    

if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    





