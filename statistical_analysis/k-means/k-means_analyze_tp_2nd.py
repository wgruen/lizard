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
import yaml
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#%matplotlib inline



palette = np.array([["red"], # index 0: red
                    ["green"], # index 1: green
                    ["blue"], # index 2: blue
                    ["black"], # index 3: black
                    ["yellow"], # index 4: yellow
                    ["purple"],
                    ["cyan"],
                    ["aqua"],
                    ["orange"],
                    ["gray"]
                    ])


def open_file_and_calculate(file_name, result, df_summary, cutoff_offset_percent):
     # open the data file
    file_path = os.path.join(os.getcwd(), file_name)
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

    df_summary = calculate(test_data, file_name, result, df_summary, cutoff_offset_percent)
    return df_summary
   

def calculate(data, filename, result, df_summary, cutoff_offset_percent):
  
    #take of the forst column
    data = data[:, 1:]
    print("data", linesep, data)

    #split first  row for X
    X = data[:, 0 ]
    print("X", linesep, X)
    

    # split Y values
    Y = data[:, 1:]
    print("Y", linesep, Y)
    print("len Y / X: ", len(Y) , len(X) )

    
    #duplicate X as many times as needed
    
    #for x in X:
    #X = [X] * len(Y[0])
    X = np.repeat(X, len(Y[0]))
    X = X.flatten()
    Y = Y.flatten()
    print("X falttened", linesep, X)
    print("Y flattened", linesep, Y)
    print("len Y / X: ", len(Y) , len(X) )


    
    # show data over X 
    plt.scatter(X, Y, s = 50, c="brown")
    plt.show()
    
    # prepare for KMeans, once array with two columns x and y
    X_kmean = np.empty([len(X), 2])
    print(X_kmean)
    X_kmean[:, 0] = Y
    X_kmean[:, 1] = Y
    print(X_kmean)
    
    
    # Seperate the data into clusters
    from sklearn.cluster import KMeans
    kmean = KMeans(n_clusters=10)
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
    
    
    ### Get the configuration
    configuration = None
    input_file_full_path = os.path.join(os.getcwd(), args.input_config_file)
    with open(input_file_full_path, 'r') as stream:
        configuration = yaml.safe_load(stream)
        
    print(yaml.safe_dump(configuration, default_flow_style=False, default_style=None))
    
    read_file_or_dir = configuration["read_file_or_dir"]
    read_full_path = os.path.join(os.getcwd(), read_file_or_dir)
    
    
    read_file_or_dir = configuration["read_file_or_dir"]
    read_full_path = os.path.join(os.getcwd(), read_file_or_dir)
    
    df_summary = pd.DataFrame(columns=["filename",\
        "# outliers",\
        "# zeros",\
        "mean",\
        "std_dev",\
        "lower_cutoff",\
        "upper_cutoff"])


    results = {}    
    # open the data file
    
    cutoff_offset_percent = configuration["cutoff_offset_percent"]
    if(os.path.isdir(read_full_path)):
        for entry in os.scandir(read_full_path):
            if entry.is_file():
                print(entry.name)
                full_filename = os.path.join(read_file_or_dir, entry.name)
                df_summary = open_file_and_calculate(full_filename,\
                        results, df_summary, cutoff_offset_percent)
                
                
    else:
        df_summary = open_file_and_calculate(read_full_path,\
                    results, df_summary, cutoff_offset_percent)
        
    
    from datetime import datetime
    now = datetime.now() # current date and time

    date_time = now.strftime("run_%m-%d-%Y--%H-%M")
    create_output_pdf(results, df_summary, date_time)
    create_output_pdf_summary(df_summary, date_time)
    
    
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
