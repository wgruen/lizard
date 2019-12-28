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
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std

from reportlab.pdfgen import canvas
from reportlab.platypus import *
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

styles = getSampleStyleSheet()


from sklearn.cluster import KMeans
#%matplotlib inline



palette = np.array([["red"], # index 0: red
                    ["green"], # index 1: green
                    ["blue"], # index 2: blue
                    ["black"], # index 3: black
                    ["yellow"], # index 4: yellow
                    ["purple"]
            
                    ])


def calculate(data, filename, result):   
    os.makedirs("output", exist_ok=True)
    pdf_out = PdfPages('output/multipage.pdf')
    
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
    print("X flattened", linesep, X)
    
    # flatten the data
    X = X.flatten()
    Y = Y.flatten()

    print("len flattened X / Y: ", len(X) , len(Y))      
    print("X flattened", linesep, X)
    print("Y flattened", linesep, Y)

      
    
    data_mean, data_std = mean(Y), std(Y)#  identify outliers
    print("Data Simple Mean:", linesep, data_mean)
    print("Data Standard Deviation:",  linesep, data_std)
    
    result[filename] = {}
    result[filename]["mean"] = data_mean
    result[filename]["std_deviaton"] = data_std
    result[filename]["X"] = X
    result[filename]["Y"] = Y
    result[filename]["title"] = filename + " - " + str(len(Y)) + " data points"
    
    # the cutoff defined as a multiple of the standard deviation
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off

    
    # identify outliers
    outliers = [x for x in Y if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    print("Outliers:", linesep, outliers)
    
    
    
    # Prepare to display outliers
    Y_outliers = Y.copy()
     # leave only the outliers in the array and display
    Y_outliers[np.logical_and(Y_outliers > lower, Y_outliers < upper)] = np.nan
    
    print("Y_outliers:", linesep, Y_outliers)
    result[filename]["outliers_Y"] = Y_outliers
    result[filename]["outliers_title"] = "Mean " + str(data_mean)\
                + "\nStd Deviation : " + str(round(data_std, 4))\
               + "\n# Outliers: " + str(len(outliers))\
               + "\nlower: " + str(round(lower, 2)) \
               + "\nupper: " + str(round(upper))
    
    
    
    #translate into X and Y
    Y_data_rows, columns =  np.shape(Y_ORI)
    
    # get the positions of outliers
    # reshape the previously flattened array
    Y_outliers.reshape((Y_data_rows, columns))
    pos_outliers = np.argwhere(~np.isnan(Y_outliers.data))
    
    print("Y_outliers reshaped: \n", len(pos_outliers))
#    Y_outliers_data_rows, Y_outliers_columns =  np.shape(Y_outliers)
 #   print("Y_outliers reshaped: \n", Y_outliers_data_rows, "  ", Y_outliers_columns)
    
    
    outliers = "\n"
    outliers_data = np.zeros(0, dtype=int) #zeros([len(pos_outliers), 4], dtype=int)
    for position in pos_outliers:
        
        #print("position: ", position)
        pos = position[0] # number of columns
        #print("pos: ", pos)
        #print("columns: ", columns)
        row = int(pos /  columns)
        #print("row: ", row)
        column = pos - row * columns
        column += 1
        #print("column: ", column)
        '''
        text = "pos, row, column, value    " + str(pos) + "\t"\
            + str(row) + "\t"\
            + str(column) + "\t"\
            + str(Y[pos])
        outliers += text
        outliers += linesep
        '''
        outliers_data = np.append(outliers_data, \
            [pos, row, column, int(Y[pos])], axis=0)
        print("outliers_data len: ", len (outliers_data))
        print("outliers_data # columns: ", outliers_data)
        #break

    #outliers_data = np.reshape(outliers_data, -1, 4)
    print("outliers:\n", outliers_data)
    result[filename]["outliers_values"] = Y_outliers.copy()
    result[filename]["outliers_data"] = outliers_data.copy()
    

    # Prepare to display zero values
    Y_zeros = Y.copy()
     # leave only the outliers in the array and display
    Y_zeros[Y_zeros != 0] = np.nan
    
    
    
    
    

def open_file_and_calcualte(file_name, result):
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

    calculate(test_data, file_name, result)
   
    
def create_output_pdf(results):
    
    pdf_out = PdfPages("output/test" + ".pdf")
    pdf_out_data = PdfPages("output/test_data" + ".pdf")


    for filename in results:
        print("filename: ", filename)
    
        plt.subplot(1, 2, 1)
        plt.title(filename + " - " + str(len(results[filename]["Y"])) + " data points")
        plt.scatter(results[filename]["X"], results[filename]["Y"], s = 50, c="brown")
    
        plt.subplot(1, 2, 2)
        plt.title("Outliers")
        plt.xlabel(results[filename]["outliers_title"])
        plt.scatter(results[filename]["X"], results[filename]["outliers_Y"], s = 50, c="blue")
        #plt.figure()
        #plot1 = plt.plot()
        plt.savefig(pdf_out, format='pdf', bbox_inches='tight')
        plt.plot()
        #break
        
     
        # print outliers to PDF
        outliers_data = results[filename]["outliers_data"]
        outliers_data = np.reshape(outliers_data, (-1, 4))
        #plt.clf()
        
        #fig, ax = plt.subplots()
        #fig.suptitle(filename + "outliers")
        #plt.title("TITLE")
        #plt.tight_layout()
    
        # hide axes
        #fig.patch.set_visible(False)
        #ax.axis('off')
        #ax.axis('tight')
        #ax.set_title(filename + " outliers")
        df = pd.DataFrame(data=outliers_data, columns=("pos", "row", "column", "value"))
        #ax.table(cellText=df.values, colLabels=df.columns, bbox=[0,0,1,1]) #loc='bottom')
        #fig.tight_layout()
        #plt.savefig(pdf_out_data, format='pdf', bbox_inches='tight')
        #plt.show()
        
        header1 = Paragraph("<para align=center>pos</para>", styles['Normal'])
        header2 = Paragraph("<para align=center>row</para>", styles['Normal'])
        header3 = Paragraph("<para align=center>colum</para>", styles['Normal'])
        header4 = Paragraph("<para align=center>value</para>", styles['Normal'])
        row_array = [header1, header2, header3, header4]
        tableHeading = [row_array]
        th = Table(tableHeading)
        
        #t1 = Table(np.array(df).tolist(), colWidths=2);
        t1 = Table(outliers_data, colWidths=2);
        doc = SimpleDocTemplate("output/table.pdf", pagesize=letter)
        element = []
        element.append(th)
        element.append(t1)
        doc.build(element)
        
        
        
    pdf_out.close()
    pdf_out_data.close()
 
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
   

    results = {}    
    # open the data file
   
    open_file_and_calcualte("../data/D2-250.csv", results)
    open_file_and_calcualte("../data/D1-150.csv", results)
    
    create_output_pdf(results)
    
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
