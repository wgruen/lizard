#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shows a simple example about how to use the K-Means algorithm
"""
#https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
#https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68

import sys
import argparse
import os
from os import linesep
import pandas as pd
import numpy as np
import pprint
import csv
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std




from sklearn.cluster import KMeans
#%matplotlib inline



palette = np.array([["red"], # index 0: red
                    ["green"], # index 1: green
                    ["blue"], # index 2: blue
                    ["black"], # index 3: black
                    ["yellow"], # index 4: yellow
                    ["purple"]
            
                    ])


def calculate(data):
    print("data", linesep, data)

    #split first  row for X
    X = data[:, 0 ]
    print("X", linesep, X)
    

    # split Y values
    Y = data[:, 1:]
    print("Y", linesep, Y)
    pprint.pprint(Y)
    
    print("len           X / Y: ", len(X) , len(Y) )
    
    #duplicate X as many times as needed
    X = np.repeat(X, len(Y[0]))
    
    # now the array have the same size and noise was removed
    #keep a copy
    X_ORI = X.copy()
    Y_ORI = Y.copy()
    
    # flatten the data
    X = X.flatten()
    Y = Y.flatten()

    print("len flattened X / Y: ", len(X) , len(Y))      
    print("X flattened", linesep, X)
    print("Y flattened", linesep, Y)

    
    # show data over X 
    plt.scatter(X, Y, s = 50, c="brown")
    plt.show()
    
    
    data_mean, data_std = mean(Y), std(Y)#  identify outliers
    print("Data Simple Mean:", linesep, data_mean)
    print("Data Standard Deviation:",  linesep, data_std)
    
    # the cutoff defined as a multiple of the standard deviation
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    
    #
    
    # identify outliers
    outliers = [x for x in Y if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    print("Outliers:", linesep, outliers)
    
    
    
    # leave only the outliers in the array and display
    Y_outliers = Y.copy()
    Y_outliers[np.logical_and(Y_outliers > lower, Y_outliers < upper)] = np.nan
    
    print("Y_outliers:", linesep, Y_outliers)
    
    #  show outliers over X 
    plt.scatter(X, Y_outliers, s = 50, c="brown")
    plt.show()
    
    # positions of outliers in the flattened array
    Y_pos_outliers = np.argwhere(~np.isnan(Y_outliers.data))
    
    #translate into X and Y
    Y_data_rows, columns =  np.shape(Y_ORI)
    for position in Y_pos_outliers:
        row = position // columns 
        column = position - row * columns
        print("row, column, value\t", int(row), "\t", int(column), "\t", int(Y[position])  )
    
    
    
    



 
'''
The parameters are self explaining

'''
def main(argv):
    parser = argparse.ArgumentParser(description='Training and running a small neural network.')
    parser.add_argument('-i', dest='input_config_file', 
                        action='store', 
                        required=True,
                        help='The yaml file containing configuration')

    args = parser.parse_args()
    #print(args)
       
    # open the data file
    file_path = os.path.join(os.getcwd(), args.input_config_file)
    test_data = ''
    print("file:", file_path)
    with open(file_path, 'r') as f:
        reader_d = csv.reader(f, delimiter=',')
        
        # get header from first row
        headers = next(reader_d)
        # get all the rows as a list
        data = list(reader_d)
        # transform data into numpy array
        test_data = np.array(data)
        

    # take of the first column, it is just empty  
    test_data = test_data[:, 1:]
    
    #convert numbers to integers
    test_data = test_data.astype(np.float) 
    #pprint.pprint(test_data)

      
    calculate(test_data)
    
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
