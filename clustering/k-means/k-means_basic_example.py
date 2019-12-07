#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shows a simple example about how to use the K-Means algorithm
"""
#https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
#https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68

from os import linesep
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#%matplotlib inline


palette = np.array([["red"], # index 0: red
                    ["green"], # index 1: green
                    ["blue"], # index 2: blue
                    ["black"], # index 3: white
                    ["yellow"], # index 4: black
                    ["purple"]
            
                    ])

# create random  data set 
# 100 rows, 2 columns
X= -2 * np.random.rand(100,2)
print("X", linesep, X)

# create a random data set of positive numbers
# 50 rows, 2 columns
X1 = 1 + 2 * np.random.rand(50,2)
print("X1", linesep, X1)

# replace line 50 to 100 of X with the values of X1
X[50:100, :] = X1

print("X first  column", linesep, X[ : , 0])
print("X second column", linesep, X[ : , 1])
print("X", linesep, X)

# show X, 
# size of marker is 20
# colour is brown
# X is the first column
# Y is the second column
# Display the data
plt.scatter(X[ : , 0], X[ :, 1], s = 50, c="brown")
plt.show()

# Seperate the data into clusters
kmean = KMeans(n_clusters=6)
kmean.fit(X)

# Find the centers of the clusters
centers = kmean.cluster_centers_
print("centers", linesep, centers)

#print the clusters and the centers
plt.scatter(X[ : , 0], X[ : , 1], s =50, c="brown")
for center in centers:
    plt.scatter(center[0], center[1], s=200, c="green", marker="s")
plt.show()

# get the associatin of data to a cluster
cluster_labels = kmean.labels_
print("cluster_labels", linesep, cluster_labels)

#print the data points belonging to a cluster
for index in range(len(cluster_labels)):
    #print(index)
    cluster_id = cluster_labels[index]
    #print(cluster_id)
    x = X[index][0]
    y = X[index][1]
    #print(x, linesep, y)
    plt.scatter(x, y, s =50, c=palette[cluster_id])
    
    
    
  
    


# predict the cluster for data point
#sample_test=np.array([-3.0,-3.0])
#second_test=sample_test.reshape(1, -1)
#Kmean.predict(second_test)