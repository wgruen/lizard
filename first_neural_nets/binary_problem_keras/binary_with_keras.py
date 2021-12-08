#!/usr/bin/env python3


# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
# https://towardsdatascience.com/a-beginners-guide-to-convolutional-neural-networks-cnns-14649dbddce8

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


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from reportlab.pdfgen import canvas
from reportlab.platypus import *
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
styles = getSampleStyleSheet()


# example of loading the cifar10 dataset
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from tensorflow import keras
from keras.regularizers import l2


class binary_with_keras():
    def __init__(self, configuration):
        ### get the parameters
        self.configuration = configuration
        self.verbose = configuration["verbose"]
        self.dropout_percentage = configuration["dropout_percentage"] / 100
        self.mkernel_regularizer_l2 = configuration["mkernel_regularizer_l2"]
        self.number_of_epochs = configuration["number_of_epochs"]
        self.machine_dump_file_name_base = configuration["machine_dump_file_name_base"]
        self.set_file_names()
        

    def set_file_names(self):
                
        self.model_file_name_base = self.machine_dump_file_name_base + \
            "-3vgg" \
            + "--epochs-" \
            + str(self.number_of_epochs) \
            + "--dropout-"  \
            + str(self.dropout_percentage) \
            + "--l2_reg-" \
            + str(self.mkernel_regularizer_l2)
            

        self.model_file_name = self.model_file_name_base + ".bin"
        self.pdf_plots_file_name  = self.model_file_name_base + "_plots.pdf"
        self.doc_summary_file_name  = self.model_file_name_base  + "_summary.pdf"
        
            
    '''
    load the dataset
    '''
    def load_dataset(self):
        # Load the training data
        file_name = self.configuration["training_data"]["training_data_file_name"]
        print("Input file name: "+ file_name)
        
        training_set_inputs = np.loadtxt(file_name)
        print("Raw training_set_inputs: ****************" + linesep + str(training_set_inputs))
        # extract input data and expected output
        self.trainY = training_set_inputs[:,-1:] # for last column
        self.trainX =  training_set_inputs[:, :-1] # for all but last column
        # Tur output into proper format
#        training_set_outputs = training_set_outputs.reshape(training_set_outputs.shape[0], 1)
        
        print("training_set_inputs: ****************"  + linesep + str(self.trainX))
        print("training_set_outputs: ****************" + linesep + str(self.trainY))
        
    
 
        # load the validation  data
        test_data = np.loadtxt(self.configuration["validation_data"])
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
        
        
        number_of_input_neurons   = self.configuration["training_data"]["number_of_input_neurons"]
        number_of_hidden_neurons  = self.configuration["training_data"]["number_of_hidden_neurons"]
        number_of_hidden_layers   = self.configuration["training_data"]["number_of_hidden_layers"]
        number_of_outputs_neurons = self.configuration["training_data"]["number_of_outputs_neurons"]

        # https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
        
        self.model = Sequential()
       
        # input shape
        # There is one data point for each input
        # and there should be three Neurons
        #
        # the first hidden layer has 4 nodes
        
        self.model.add(Dense(6, input_dim=3, activation='relu'))
        
        # The hidden layer shall have 4 Neurons
        # one hidden layer should be sufficient for the 
        # majority of problem :-)
        # Hidden layers are required only, and only if the data 
        # must be separated non-linearly
        # self.model.add(Dense(4, activation='relu'))
        
        # The output layer should be one Neuron
        self.model.add(Dense(1, activation='sigmoid'))
     
        # pick an optimizer for gradient descent with momentum optimizer
        #opt = SGD(lr=0.001, momentum=0.9)
        
        # compile the model
        self.compile_output = self.model.compile(optimizer='SGD', loss='binary_crossentropy') #, metrics=['accuracy'])
        
        
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
     
        self.fit_history = self.model.fit(self.trainX, self.trainY,\
            epochs=self.number_of_epochs,\
            batch_size=1,\
            validation_data=(self.testX, self.testY),\
            verbose=self.verbose)
    

        # The history dictionary will have these keys: 
        # ['val_loss', 'val_acc', 'loss', 'acc']
            
       

        # SAVE the trained model
        self.model.save(self.model_file_name)
        self.create_output_pdf()
        
        
            
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
        #_, acc = 
        self.run_history = self.model.evaluate(self.testX, self.testY, verbose=self.verbose)
        print(self.run_history)
        
    '''
    The model is evaluated with the test data, which is part of
    the training data
    '''
    def predict_data(self):
        ### load and prepare the dataset
        self.load_dataset()
        self.print_dataset()
        
        
        
        ## load a previously stored model
        self.model = keras.models.load_model(self.model_file_name)
    
    
        ## EVALUATE the previously trained model
        self.testX = self.prepare_data(self.testX)
        #_, acc = 
        self.run_history = self.model.predict(self.testX, verbose=self.verbose)
        print(self.run_history)
        


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
            
    if(configuration["evaluate_the_model"] == 1):
        mymodel = binary_with_keras(configuration)
        mymodel.predict_data()

    

if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    





