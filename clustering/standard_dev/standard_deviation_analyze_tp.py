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
  
    #take of the first column
    data = data[:, 1:]
    print("data", linesep, data)

    #split first  row for X
    X = data[:, 0 ]
    print("X", linesep, X)
    

    # split Y values
    Y = data[:, 1:]
    print("Y", linesep, Y)
    pprint.pprint(Y)
    
    print("len Y / X: ", len(Y) , len(X) )
    print("len Y columns / X columns: ", len(Y[0]) , len(X[0]) )

    
    #duplicate X as many times as needed
    
    #for x in X:
    #X = [X] * len(Y[0])
    X = np.repeat(X, len(Y[0]))
    X = X.flatten()
    Y1 = Y.flatten()
    Y = Y1
    print("X flattened", linesep, X)
    print("Y1 flattened", linesep, Y)
    print("len Y / X: ", len(Y) , len(X) )
    print("len Y columns / X columns: ", len(Y[0]) , len(X[0]) )
    

    
    # show data over X 
    plt.scatter(X, Y, s = 50, c="brown")
    plt.show()
    
    
    data_mean, data_std = mean(Y), std(Y)# identify outliers
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    
    # identify outliers
    outliers = [x for x in Y if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers)) 
    return

    # prepare for KMeans, once array with two columns x and y
    X_kmean = np.empty([len(X), 2])
    print(X_kmean)
    X_kmean[:, 0] = X
    X_kmean[:, 1] = Y
    print(X_kmean)
    
    
    # Seperate teh data into clusters
    from sklearn.cluster import KMeans
    kmean = KMeans(n_clusters=2)
    kmean.fit(X_kmean)
    
    # Find the centers of the clusters
    centers = kmean.cluster_centers_
    print("centers", linesep, centers)
    
    #print the clusters and the centers
    plt.scatter(X_kmean[ : , 0], X_kmean[ : , 1], s =50, c="brown")
    for center in centers:
        plt.scatter(center[0], center[1], s=200, c="green", marker="s")
    plt.show()
    
    # get the associatin of data to a cluster
    cluster_labels = kmean.labels_
    #print("cluster_labels", linesep, cluster_labels)
    
    #print the data points belonging to a cluster
    for index in range(len(cluster_labels)):
        #print("index: ", index)
        cluster_id = cluster_labels[index]
        #print("clustere_id: ", cluster_id)
        x = X_kmean[index][0]
        y = X_kmean[index][1]
        #print(x, linesep, y)
        plt.scatter(x, y, s =50, c=palette[cluster_id])
        
        

 
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
        reader = csv.reader(f, delimiter=',')
        # get header from first row
        headers = next(reader)
        # get all the rows as a list
        data = list(reader)
        # transform data into numpy array
        test_data = np.array(data)
        
    pprint.pprint(test_data)
        
      
    calculate(test_data)
    
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
