#!/usr/bin/env python3

import sys
import argparse
import os
import inspect
import importlib
from numpy import exp, array, random, dot, round
import numpy as np
from os import linesep
import pprint, pickle
import json
import yaml


    
def main(argv):
    #Read the confiuration YAML file
    with open("example_1.configuration.yaml", 'r') as stream:
        configuration = yaml.safe_load(stream)

    parser = argparse.ArgumentParser(description='Training and running a small neural network.')
    parser.add_argument('-i', dest='input_config_file', action='store', 
                        help='the yaml file containing confiiguration')

    args = parser.parse_args()
    print(args)

   
   #print 'Input file is "', inputfile

    
    learn(configuration)
    calculate(configuration)
    
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
