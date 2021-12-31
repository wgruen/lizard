#!/usr/bin/env python3


import sys
import argparse
import os
import inspect
import importlib
import json
import yaml
import binary_with_keras
import numpy as np
import keras
import gc
import pprint
from time import gmtime, strftime

    
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
    
    # constructor needs these, but they are not needed for prediction
    # or evaluation
    configuration["mkernel_regularizer_l2"] = configuration["mkernel_regularizer_l2_range"][0]
    configuration["dropout_percentage"] = configuration["dropout_percentage_range"][0]
      
    result = ""
    result_summary = ""
    # on log file for the entire run
    dt = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    dt = "---" + dt + ".log"
    
    hn_layers_1_range = configuration["machine"]["number_of_hidden_neurons_layers_1_range"]
    for nl1 in hn_layers_1_range:
        configuration["machine"]["number_of_hidden_neurons_layers_1"] = nl1  
        hn_layers_2_range = configuration["machine"]["number_of_hidden_neurons_layers_2_range"]
        
        
        for nl2 in hn_layers_2_range:
            configuration["machine"]["number_of_hidden_neurons_layers_2"] = nl2
            mymodel = binary_with_keras.binary_with_keras(configuration)
            res, ress = mymodel.predict_data()
            result += res
            result_summary += ress
            
            if not os.path.exists("logs"):
                os.mkdir("logs")
            file_name = mymodel.logfile_name
            model_file_name_and_path = os.path.join("logs", file_name + "-predict" + dt)
            with open(model_file_name_and_path, 'w') as fd:
                fd.write(result)
                fd.write("\n")
                
                content = yaml.dump(configuration)
                fd.write(content)
                
            model_file_name_and_path = os.path.join("logs", file_name + "-predict_summary" + dt)
            with open(model_file_name_and_path, 'w') as fd:
                fd.write(result_summary)
                fd.write("\n")
                
                #content = yaml.dump(configuration)
                #fd.write(content)
                
                
            keras.backend.clear_session()
            gc.collect()     
            del mymodel
            
            
    print(result_summary)              

    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
