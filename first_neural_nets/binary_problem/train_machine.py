#!/usr/bin/env python3

import sys
import argparse
import os
import inspect
import importlib
import json
import yaml
import binary_problem as bp


    
def main(argv):

    parser = argparse.ArgumentParser(description='Training and running a small neural network.')
    parser.add_argument('-i', dest='input_config_file', action='store', 
                        help='the yaml file containing configuration')

    args = parser.parse_args()
    print(args)

    #Read the confiuration YAML file
    with open(args.input_config_file, 'r') as stream:
        configuration = yaml.safe_load(stream)
    
    bp.learn(configuration)
    
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
