#!/usr/bin/env python3

import sys
import argparse
import os
import inspect
import importlib
import json
import yaml
import binary_with_keras


    
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
    
    if(configuration["fit_the_model"] is 1):
        for kernel_regulror_l1l2 in configuration["mkernel_regularizer_l2_range"]:
            configuration["mkernel_regularizer_l2"] = kernel_regulror_l1l2
            for dropout in configuration["dropout_percentage_range"]:
                configuration["dropout_percentage"] = dropout
                mymodel = binary_with_keras.binary_with_keras(configuration)
                mymodel.fit_model()
    
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
