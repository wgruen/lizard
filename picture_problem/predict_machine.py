#!/usr/bin/env python3

import sys
import argparse
import os
import inspect
import importlib
import json
import yaml
from picture_3vgg_layers import myCIFAR10
import numpy as np
import keras
import gc
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
    
    results = np.empty((0,2), str)
    
    # on log file for the entire run
    dt = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    dt = "---" + dt
    
    for kernel_regulror_l1l2 in configuration["mkernel_regularizer_l2_range"]:
        configuration["mkernel_regularizer_l2"] = kernel_regulror_l1l2
        for dropout in configuration["dropout_percentage_range"]:
            configuration["dropout_percentage"] = dropout
            
            mymodel = myCIFAR10(configuration)
            total_error = mymodel.predict_data()
            results = np.append(results, np.array([mymodel.model_file_name_base, str(total_error)]))
            print(results)
                    
                    
                    
            if not os.path.exists("logs"):
                os.mkdir("logs")
            file_name = mymodel.logfile_name
            model_file_name_and_path = os.path.join("logs", file_name + dt)
            with open(model_file_name_and_path, 'w') as fd:
                content = str(results)
                fd.write(content)
                fd.write("\n")
                        
                content = yaml.dump(configuration)
                fd.write(content)
                        
                        
            keras.backend.clear_session()
            gc.collect()     
            del mymodel
                      
                    
                 
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
