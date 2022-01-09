#!/usr/bin/env python3

import sys
import argparse
import os
import inspect
import importlib
import json
import yaml
import gzip
import binary_with_keras
import numpy as np
from tensorflow import keras as keras
import gc
from time import gmtime, strftime
import pandas as pd



    
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
        
    #print(yaml.safe_dump(configuration, default_flow_style=False, default_style=None)) 
    
    result_details = ""
    result_summary = pd.DataFrame()

    # on log file for the entire run
    dt = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    dt = "---" + dt + ".log"
    
    for num_neurons_in_layer_1 in configuration["machine"]["number_of_hidden_neurons_layers_1_range"]:
        configuration["machine"]["number_of_hidden_neurons_layers_1"] = num_neurons_in_layer_1
        
        for num_neurons_in_layer_2 in configuration["machine"]["number_of_hidden_neurons_layers_2_range"]:
            configuration["machine"]["number_of_hidden_neurons_layers_2"] = num_neurons_in_layer_2
            
            for lrate in configuration["machine"]["learning_rate_range"]:
                configuration["machine"]["learning_rate"] = lrate
                    
                for batch_size in configuration["machine"]["batch_size_range"]:
                    configuration["machine"]["batch_size"] = batch_size
                    
                    for epochs in configuration["machine"]["number_of_epochs_range"]:
                        configuration["machine"]["number_of_epochs"] = epochs
        
                        mymodel = binary_with_keras.binary_with_keras(configuration)
                        training_data = mymodel.fit_model()
                 
                        if not os.path.exists("logs"):
                            os.mkdir("logs")

                        # result details
                        file_name = mymodel.logfile_name
                        model_file_name_and_path = os.path.join("logs", file_name + "-train" + dt)
                        with open(model_file_name_and_path, 'a') as fd:
                            fd.write(result_details)
                            fd.write("*** Next machine ****" + os.linesep)
                 
                            content = yaml.dump(configuration)
                            fd.write(content + os.linesep)
                       
                        # predict summary
                        predict_delta = mymodel.predict_data()


                        #print(predict_delta)
                        training_data["predict"] = predict_delta["predict"]
                        #print(training_data)
                        training_data = pd.DataFrame([training_data])
                        #print(training_data)
                        result_summary = result_summary.append(training_data) 
                        #print(result_summary)
                       
                        
                        model_file_name_and_path = os.path.join("logs", file_name + "-train-summary" + dt)
                        with open(model_file_name_and_path, 'w') as fd:
                            for col_name in result_summary.columns: 
                                fd.write("Next sorted by column: " + col_name + os.linesep)
                                result_summary_by_val_accuracy = result_summary.sort_values(by=col_name)
                                dfAsString = result_summary.to_string()
                                fd.write(dfAsString)
                                fd.write(os.linesep + os.linesep)



                        # detailed logs 
                        model_file_name_and_path = os.path.join("logs", file_name + "-train-detailed-logs" + dt) # + ".gzip")
                        with open(model_file_name_and_path, 'a') as fd:
                        #with gzip.GzipFile(model_file_name_and_path, 'w') as fd:
                            #compressed = zlib.compress(mymodel.logging)
                            fd.write(mymodel.logging)

                      
                 
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
