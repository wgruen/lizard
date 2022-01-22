#!/usr/bin/env python3

from re import I
import sys
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
import pandas as pd
import decimal
import string
from time import gmtime, strftime
import shutil

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from reportlab.pdfgen import canvas
from reportlab.platypus import *
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
styles = getSampleStyleSheet()


# example of loading the cifar10 dataset
import gc
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
#from keras import optimizers 
from tensorflow.keras.callbacks import History, EarlyStopping, CSVLogger, LearningRateScheduler


change_every_epochs = 100
change_divide_by = 10

def lr_scheduler(epoch, lr):
    #print(change_every_epochs)
    if epoch is not 0 and epoch is change_every_epochs:
        print("change at epoch:" + str(epoch) + os.linesep)
        return lr / change_divide_by

    return lr

class binary_with_keras():
    def __init__(self, configuration):
        ### get the parameters
        self.configuration = configuration
        self.verbose_logging = configuration["general"]["verbose_logging"]
        self.verbose = configuration["machine"]["verbose"]
        #self.dropout_percentage = configuration["dropout_percentage"] / 100
        #self.mkernel_regularizer_l2 = configuration["mkernel_regularizer_l2"]
        self.number_of_epochs = configuration["machine"]["number_of_epochs"]
        self.learning_rate = configuration["machine"]["learning_rate"]
        self.batch_size = configuration["machine"]["batch_size"]
        self.loss_function  = self.configuration["machine"]["loss_function"] 
        
        self.machine_dump_file_name_base = configuration["machine"]["machine_dump_file_name_base"]
        self.set_file_names()
        self.date_and_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.date_and_time = "---" + self.date_and_time
        self.logfile_name = self.machine_dump_file_name_base + ".log"
        self.model_file_name_and_path = os.path.join("saved_machines", self.model_file_name)
        self.logging = ""

        # cleanup happens here
        tf.keras.backend.clear_session()
        gc.collect() 

        tf.get_logger().setLevel('WARNING')

        global change_every_epochs
        global change_divide_by
        change_every_epochs = self.configuration["learning_rate_scheduler"]["change_every_epochs"]
        change_divide_by = self.configuration["learning_rate_scheduler"]["change_divide_by"]

        if not os.path.exists("output"):
          os.mkdir("output")

        self.use_tpu = False
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
            #strategy = tf.distribute.TPUStrategy(resolver)


            tf.config.list_physical_devices('TPU') 
            tf.tpu.experimental.initialize_tpu_system(resolver)
            print("All devices: ", tf.config.list_logical_devices('TPU'))
            self.use_tpu = True

        except:
            pass
            print("not connected to a TPU")
    

    def __del__(self):
        if hasattr(self, "model") is True:
            del self.model
        if hasattr(self, "my_model_predict") is True:
            del self.my_model_predict
        
        tf.keras.backend.clear_session()
        gc.collect()     

    def set_file_names(self):
        self.model_file_name_base = "".join(( self.machine_dump_file_name_base,
            "--ep-",
            str(self.number_of_epochs), 
            "--i-",
            str(self.configuration["machine"]["number_of_input_neurons"]),
            "--hl1-",
            str(self.configuration["machine"]["number_of_hidden_neurons_layers_1"]),
            "--hl2-",
            str(self.configuration["machine"]["number_of_hidden_neurons_layers_2"]),
            "--lr-",
            str(self.learning_rate),
            "--bs-",
            str(self.batch_size)
            ))
             
 
        self.model_file_name = self.model_file_name_base + ".bin"
        self.pdf_plots_file_name  = self.model_file_name_base + "_plots"
        self.doc_summary_file_name  = self.model_file_name_base  + "_summary"
        

    def log_level(self, log_string : str):
        if self.verbose_logging:
            print(log_string)
        else:
            self.logging += str(log_string)
    
            
    '''
    load the dataset
    '''
    def load_dataset(self):
        # Load the training data
        self.log_level("=== enter load_dataset ===" + os.linesep)
        training_set_inputs = []
        
        if "training_data_file_name" in self.configuration["training_data"]:
            # load from file
            file_name = self.configuration["training_data"]["training_data_file_name"]
            self.log_level("Input file name: "+ file_name + os.linesep)
            training_set_inputs = np.loadtxt(file_name)
        else:
            training_data_set = self.configuration["training_data"]["train_with_embedded"]         
            training_set_inputs = self.configuration["training_data"][training_data_set]
            training_set_inputs = np.array(training_set_inputs)
        
        # prepare for printing
        self.training_set_inputs_df = pd.DataFrame(training_set_inputs)
        
        # extract input data and expected output
        self.trainY = training_set_inputs[:,-1:] # for last column
        self.trainX =  training_set_inputs[:, :-1] # for all but last column
    
        # load the validation  data
        if "validation_data_file_name" in self.configuration:
            test_data = np.loadtxt(self.configuration["validation_data_file_name"])
        else:
            test_data = self.configuration["validation_data_embedded"]
            test_data = np.array(test_data)

            
        #test_data = np.array(test_data, dtype=np.dtype(decimal.Decimal))
        self.testY = test_data[:,-1:] # for last column
        self.testX = test_data[:, :-1] # for all but last column 

        #self.print_dataset()


    def print_dataset(self):
        # summarize loaded dataset
        self.log_level("=== enter print_dataset ===" + os.linesep)
        self.log_level(str("self.trainX") + os.linesep + str(self.trainX) + os.linesep)
        self.log_level(str("self.trainY") + os.linesep + str(self.trainY) + os.linesep)
        self.log_level(str("self.testX") + os.linesep + str(self.testX) + os.linesep)
        self.log_level(str("self.testY") + os.linesep + str(self.testY) + os.linesep)

    
    
    '''
    The integers are in the range of 0 to 255
    Here we normalize the picture values.
    Rescale them to the range [0,1]
    
    This prepares the pixel data
    '''
    def prepare_data(self, data):
        # convert from integers to floats
        
        # normalize to range 0-1
 #       data_norm = data_norm / 255.0

        return data
 
       
    '''
    Define the keras model.
    Please note that the model defintion is highly dependenet on the 
    image size
    
    Here we build the layers of the CNN (Convolutional Neural Networks) model
    CCN was designed to work with two dimensional image data.
    '''
    def define_model(self):
        self.log_level("=== enter define model ===" + os.linesep)
        number_of_input_neurons   = self.configuration["machine"]["number_of_input_neurons"]
        number_of_hidden_neurons_layers_1  = self.configuration["machine"]["number_of_hidden_neurons_layers_1"]
        number_of_hidden_neurons_layers_2   = self.configuration["machine"]["number_of_hidden_neurons_layers_2"]
        number_of_outputs_neurons = self.configuration["machine"]["number_of_outputs_neurons"]
        my_bias_initializer = self.configuration["machine"]["bias_initializer"] 
        my_metrics = self.configuration["machine"]["metrics"] 
        my_optimizer = self.configuration["machine"]["optimizer"] 
    
        
        # https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
        
        self.model = Sequential()
       
        # input shape
        # There is one data point for each input
        
        self.model.add(Dense(
            number_of_hidden_neurons_layers_1,
            input_dim=number_of_input_neurons,
            activation='relu',
            bias_initializer=my_bias_initializer))
        
        # One hidden layer should be sufficient for the 
        # majority of problems :-)
        # Hidden layers are required only, and only if the data 
        # must be separated non-linearly
        if number_of_hidden_neurons_layers_2 is not 0:
            self.model.add(Dense(
                number_of_hidden_neurons_layers_2, 
                activation='relu',
                bias_initializer=my_bias_initializer))
        
        # The output layer should be one Neuron
        self.model.add(Dense(number_of_outputs_neurons, activation='sigmoid'))

        # pick an optimizer for gradient descent with momentum optimizer

        if my_optimizer == 'SGD':
            self.opt = SGD(learning_rate=self.learning_rate, momentum=0.9)
        
        if my_optimizer == 'adam':
            self.opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        self.log_level("optimizer: " + str(self.opt) + os.linesep)


        # compile the model
        #https://keras.io/api/metrics/
        self.compile_output = self.model.compile(run_eagerly=True, optimizer=self.opt, loss=self.loss_function,  metrics=[my_metrics]) 
        
        self.log_level(self.model.summary()) 
   
     
    
    '''
    The model will be trained (fit) and
    the trained model will be stored to hd.
    
    '''
    def fit_model(self):
       
        # 
        if not os.path.exists("saved_machines"):
            os.mkdir("saved_machines")

        # don't overwrite a model, delete the old model first
        #print(self.model_file_name_and_path)
        shutil.rmtree(self.model_file_name_and_path, ignore_errors=True)   
       
        ### load and prepare the dataset
        self.log_level("=== enter fit model ===" + os.linesep)
        self.load_dataset()
        
        # normalize the pixel data for X only
        self.trainX = self.prepare_data(self.trainX)
        self.testX = self.prepare_data(self.testX)

        self.print_dataset()
        
        ### Define and use a model
        self.define_model()
        
        ### FIT / TRAIN the model
        # https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/

        history = History()

        csv_file_name_and_path = os.path.join("output", self.pdf_plots_file_name + self.date_and_time + ".csv")
        #csv_log = CSVLogger(csv_file_name_and_path)
        
        early_stopping = EarlyStopping(
            monitor=self.configuration["early_stopping"]["monitor"] ,
            patience=self.configuration["early_stopping"]["patience"] ,
            verbose=self.configuration["early_stopping"]["verbose"]
            #  min_delta=self.configuration["early_stopping"]["min_delta"]
        )

        learning_rate_cb = LearningRateScheduler(lr_scheduler)

        callbacks = [
            history, 
            #csv_log,
            early_stopping,
            learning_rate_cb
        ]

        if self.use_tpu is False:
            callbacks.append([CSVLogger(csv_file_name_and_path)])


        trainx_tensor = tf.constant(self.trainX)
        trainy_tensor = tf.constant(self.trainY)
        testx_tensor = tf.constant(self.testX)
        testy_tensor = tf.constant(self.testY)


        #print("after making it a Tensor")
        #print(trainx_tensor)
        #print(trainy_tensor)
        #print(testx_tensor)
        #print(testy_tensor)
        #print(self.batch_size)
        #print(self.number_of_epochs)
        
        self.fit_history = self.model.fit(trainx_tensor, trainy_tensor,\
            epochs=self.number_of_epochs,\
            batch_size=self.batch_size,\
            validation_data=(testx_tensor, testy_tensor),\
            verbose=self.verbose, \
            callbacks=callbacks)
          
        if self.use_tpu is False:
            tf.keras.models.save_model(self.model, self.model_file_name_and_path)

        # for binary_crossentropy
        #['loss', 'accuracy', 'val_loss', 'val_accuracy'])
        #print(history.history)
        #self.log_level(str(self.model.history.history) + os.linesep)
        self.epochs = 0
        if "loss"  in history.history:
            self.epochs = len(history.history['loss'])

        training_data = {
            "machine": self.model_file_name_and_path, \
            "epochs" : self.epochs, \
        }

        for key in self.model.history.history:
             # tale care of long key strings
             dict_key = key
             if key is "mean_squared_logarithmic_error":
                dict_key = "msle"
             if key is "val_mean_squared_logarithmic_error":
                dict_key = "val_msle"

             training_data[dict_key] = round(self.fit_history.history[key][-1], 6)

       
        self.log_level(json.dumps(training_data, indent=4) + os.linesep)
        
        self.create_output_pdf()

        return training_data
            
    def create_output_pdf(self):
        ##############################################
        ### create a PDF with plots
        ##############################################

        if self.use_tpu is True:
            return
            
        file_name_and_path = os.path.join("output", self.pdf_plots_file_name + self.date_and_time + ".pdf")
        pdf_plots  = PdfPages(file_name_and_path)

        figs = plt.figure()
        fig = plt.figure() #figsize=(10, 10))

        
        # SAVE history  of the trained model 
        #self.log_level("fit_history keys: " + str(self.fit_history.history.keys()))
        #self.log_level("fit_history" + str(self.fit_history.history))
        
         
        plot_num = 321
        
        if 'loss' in self.fit_history.history:
            plt.subplot(plot_num)
            plot_num += 1
            plt.title('Cross Entropy Loss')
            plt.ylabel('Loss')
            plt.xlabel('epoch')
            #plt.legend(['train', 'test']) 
            #plt.figure(figsize=(10, 10)) # inches
         
            plt.plot(self.fit_history.history['loss'], color='blue', label='train model')
            plt.plot(self.fit_history.history['val_loss'], color='orange', label='validate model')
            plt.legend(loc='best')
      
        
        # plot accuracy
        if 'auc' in self.fit_history.history:
            plt.subplot(plot_num)
            plot_num += 1            
            plt.title('Classification Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('epoch')
            #plt.legend(['train', 'test'])
            #plt.figure(figsize=(10, 10)) # inches
          
            plt.plot(self.fit_history.history['auc'], color='blue', label='train_model')
            plt.plot(self.fit_history.history['val_auc'], color='orange', label='validate_model')
            plt.legend(loc='best')


        # plot accuracy
        if 'accuracy' in self.fit_history.history:
            plt.subplot(plot_num)
            plot_num += 1
            plt.title('Classification Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('epoch')
            #plt.legend(['train', 'test']')
            #plt.figure(figsize=(10, 10)) # inches
           
            plt.plot(self.fit_history.history['accuracy'], color='blue', label='train_model')
            plt.plot(self.fit_history.history['val_accuracy'], color='orange', label='validate_model')
            plt.legend(loc='best')            
        
       # plot accuracy
        if 'mean_squared_logarithmic_error' in self.fit_history.history:
            plt.subplot(plot_num)
            plot_num += 1
            plt.title('mean_squared_logarithmic_error')
            plt.ylabel('Error')
            plt.xlabel('epoch')
            #plt.legend(['train', 'test'], loc='upper left')
            #plt.figure(figsize=(10, 10)) # inches
            plt.plot(self.fit_history.history['mean_squared_logarithmic_error'], color='blue', label='train_model')
            plt.plot(self.fit_history.history['val_mean_squared_logarithmic_error'], color='orange', label='validate_model')
            plt.legend(loc='best')         
         
        
        # save plot to file
        plt.tight_layout() #pad=0.4, w_pad=0.5, h_pad=1.0)
        pdf_plots.savefig(fig) #, bbox_inches='tight')
        pdf_plots.close()
        
        
        ##############################################
        ### create a PDF with the machine details
        ##############################################
        doc_summary  = SimpleDocTemplate("output/" + self.doc_summary_file_name  + self.date_and_time + ".pdf", pagesize=letter)
    
        element = []
        header = Paragraph("\nSummary of Training Runs", styles["Heading1"])
        element.append(header)
        
        header = Paragraph("\nThe script's input parameters", styles["Heading2"])
        element.append(header)
        text = yaml.dump(self.configuration, indent=4)
        self.log_level(text)
        #pp = pprint.PrettyPrinter(indent=4)
        #text = pp.pprint(text)
        para = XPreformatted(text, styles["Code"], dedent=0)
        element.append(para)

        header = Paragraph("\nThe class / runs parameters", styles["Heading2"])
        element.append(header)
        text =  "number of epochs provided to fit: " + str(self.number_of_epochs) + os.linesep
        text +=  "number of epochs the machine ran: " + str(self.epochs) + os.linesep
        self.log_level(text)
        para = XPreformatted(text, styles["Code"], dedent=0)
        element.append(para)


                
        header = Paragraph("\nThe model summary", styles["Heading2"])
        element.append(header)
        f = io.StringIO() 
        with redirect_stdout(f):
            self.model.summary() 
        s = f.getvalue()
        #self.log_level("model summary:\n", s)
        para = XPreformatted(s, styles["Code"], dedent=0)
        element.append(para)  
        

        header = Paragraph("\nThe model metric", styles["Heading2"])
        element.append(header)
        f = io.StringIO() 
        with redirect_stdout(f):
            self.model.metrics_names
        s = f.getvalue()
        #self.log_level("metric names:\n", s)
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

        
        ## load a previously stored model
        self.model = tf.keras.models.load_model(self.model_file_name_and_path)
    
        ## EVALUATE the previously trained model
        self.testX = self.prepare_data(self.testX)
        self.eval = self.model.evaluate(self.testX, self.testY, verbose=self.verbose)
        
        #self.log_level("=== after eval ===")
        #self.log_level(self.testX)
        #self.log_level(self.testY)
        #self.log_level(self.model.metrics_names)
        #self.log_level("result below")
        #self.log_level(self.eval)
        
    '''
    The model is evaluated with the test data, which is part of
    the training data
    '''
    def predict_data(self):
        ### load and prepare the dataset
        self.log_level("=== enter predcit data ===" + os.linesep)
        self.load_dataset()
        
        ## PREDICT with the previously trained model
        self.testX = self.prepare_data(self.testX)
        self.testY = self.prepare_data(self.testY)

        #self.print_dataset()
        testx_tensor = tf.constant(self.testX)
        #testy_tensor = tf.constant(self.testY)

        output_details = "==================== new machine run ====================" + os.linesep
        output_details += self.model_file_name_base + os.linesep
        predict_summary = self.model_file_name_base
        
        if self.use_tpu is False:
            ## load a previously stored model
            if not os.path.exists(self.model_file_name_and_path):
                output_details = "=== can't load machine - the file is not available" + os.linesep
                output_details += self.model_file_name_and_path + os.linesep
                return_value = { 
                    "predict_details": output_details, 
                    "predict_summary": output_details
                }

                return return_value

       
            self.my_model_predict = tf.keras.models.load_model(self.model_file_name_and_path)
        else:
            self.my_model_predict = self.model
        
        pred = self.my_model_predict.predict(testx_tensor, verbose=self.verbose)
        
           
        output_details += "=== after prediction ===" + os.linesep
        output_details += np.array_str(pred, precision=6, suppress_small=True) + os.linesep
 
        #print(output_details)

        # difference to expectations
        delta = np.absolute(pred - self.testY)
        output_details = "=== the difference to the expected value ===" + os.linesep
        output_details += str(delta) + os.linesep

        #print(output_details)

        # show them side by side
        side_by_side = np.dstack((pred, delta))
        output_details += "=== predicted , predicted - expected ===" + os.linesep
        output_details += str(side_by_side) + os.linesep

        #print(output_details)        

        total_error_sum = sum(delta)

        output_details += "=== the sum of the total error ===" + os.linesep
        output_details += str(total_error_sum) + os.linesep
        
        #print(output_details)

        predict_summary += "\ttotal_error_sum: \t" + str(total_error_sum) + os.linesep 

        return_value = {
            "predict":  round(float(total_error_sum[0]), 6), 
            "predict_details": output_details,  
            "delta": delta, 
            "predict_summary": predict_summary
        }


       # print(return_value)    
        return return_value
        


'''
main
'''
def main(argv):
    print("nothing to do here")

    

if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    





