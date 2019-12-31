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
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
#from sklearn.cluster import KMeans
from numpy import percentile

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



def get_table_for_special_positions(pos_specials, columns, Y_data):
    specials = "\n"
    specials_data = np.zeros(0, dtype=int) 
    for position in pos_specials:
        
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
        specials += text
        specials += linesep
        '''
        specials_data = np.append(specials_data, \
            [pos, row, column, int(Y_data[pos])], axis=0)
        #print("outliers_data len: ", len (outliers_data))
        #print("outliers_data # columns: ", outliers_data)
        #break
        
    return specials_data


def calculate(data, filename, result, df_summary, cutoff_offset_percent):   
    os.makedirs("output", exist_ok=True)
   # pdf_out = PdfPages('output/multipage.pdf')
    
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
    
    ### Get the data mean and std deviation
    result[filename] = {}
    result[filename]["X"] = X
    result[filename]["Y"] = Y
    result[filename]["title"] = filename + " - " + str(len(Y)) + " data points"
    
    
    # for interquartile, supppsed to be 25 and 75
    q_low_percentile  = 50 - cutoff_offset_percent
    q_high_percentile = 50 + cutoff_offset_percent
    print("q_low_percentile: ", q_low_percentile)
    print("q_high: ", q_high_percentile)
    
    
    quartile_low, quartile_high = percentile(Y, q_low_percentile), percentile(Y, q_high_percentile)
    print("quartile_low: ", quartile_low)
    print("quartile_high: ", quartile_high)
    
    # interquartile range - IQR
    # difference between upper and lower quartiles
    interquartile_range = quartile_high - quartile_low
    print("interquartile_range: ", interquartile_range)
            

    '''
    title = 'Percentiles: %ith=%.3f,\n %ith=%.3f,\n interquartile range=%.3f' \
          % (q_low_percentile, quartile_low,\
             q_high_percentile, quartile_high,\
             interquartile_range)
    '''
    
    title = "Percentiles: \n" +\
        str(q_low_percentile) + "th=" + str(quartile_low) + "\n" +\
        str(q_high_percentile) + "th=" + str(quartile_high) + "\n" +\
        "interquartile range: " + str(interquartile_range)   
    print(title)
    
    
    # calculate the outlier cutoff
    # Outliers are often defined as values that fall below 1.5 * IQR
    cut_off = interquartile_range * 1.5
    lower, upper = quartile_low - cut_off, quartile_high + cut_off
    # identify outliers
    outliers = [x for x in Y if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    # remove outliers
    outliers_removed = [x for x in Y if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(outliers_removed))
    
    print('Identified outliers: %d' % len(outliers))
    print("Outliers:", linesep, outliers)
    
    
    
    ### Prepare to display outliers
    Y_outliers = Y.copy()
     # leave only the outliers in the array and display
    Y_outliers[np.logical_and(Y_outliers > lower, Y_outliers < upper)] = np.nan
    
    print("Y_outliers:", linesep, Y_outliers)
    result[filename]["outliers_Y"] = Y_outliers
    title = title\
         + "\n# Outliers: " + str(len(outliers))\
         + "\nlower: " + str(round(lower, 2)) \
         + "\nupper: " + str(round(upper, 2))
    
    result[filename]["outliers_title"] = title
    
    
    # get the number of rows and columsn of the original data
    Y_data_rows, columns =  np.shape(Y_ORI)
    
    ### get the positions of outliers
    # reshape the previously flattened array
    Y_outliers.reshape((Y_data_rows, columns))
    pos_outliers = np.argwhere(~np.isnan(Y_outliers.data))    
    print("len pos_outliers: \n", len(pos_outliers))   
    
    outliers_data = get_table_for_special_positions(\
                pos_outliers, columns, Y)

        
    #print("outliers:\n", outliers_data)
    result[filename]["outliers_values"] = Y_outliers.copy()
    result[filename]["outliers_data"] = outliers_data.copy()
    

    ### Get the position of  zero values
    Y_zeros = Y.copy()
     # leave only the outliers in the array and display
    Y_zeros[Y_zeros != 0] = np.nan
    Y_zeros.reshape((Y_data_rows, columns))
    pos_zeros = np.argwhere(~np.isnan(Y_zeros.data))    
    print("len pos_zeros: \n", len(pos_zeros))   
    
    zeros_data = get_table_for_special_positions(\
                pos_zeros, columns, Y)
        
    #print("outliers:\n", outliers_data)
    result[filename]["zeros_data"] = zeros_data.copy()
  
    '''
       title = "Percentiles: \n" +\
        str(q_low_percentile) + "th=" + str(quartile_low) + "\n" +\
        str(q_high_percentile) + "th=" + str(quartile_high) + "\n" +\
        "interquartile range: " + str(interquartile_range)
 
       title = title\
         + "\n# Outliers: " + str(len(outliers))\
         + "\nlower: " + str(round(lower, 2)) \
         + "\nupper: " + str(round(upper, 2))
         '''
    
    
    summary_row1 = pd.Series([filename,\
            len(pos_outliers),\
            len(pos_zeros),\
            round(quartile_low, 2),\
            round(quartile_high, 2),\
            round(interquartile_range, 2),\
            round(lower, 2),\
            round(upper, 2)],\
            index=df_summary.columns)
    
    df_summary = df_summary.append(summary_row1, ignore_index=True)
    print("df_summary: \n", df_summary)
    return df_summary
    

def open_file_and_calcualte(file_name, result, df_summary, cutoff_offset_percent):
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
   
    
def create_output_pdf_for_special_values(special_data, element, filename):
        # print special data to PDF
        
        header = Paragraph(filename, styles["Heading1"])
        element.append(header)
        
        special_data = special_data.reshape(-1, 4) 
        print("special data: ", special_data)
        
        # get number of rows
        rows, coluums = special_data.shape
        txt = Paragraph("Number of findings: " + str(rows), styles["Normal"])
        element.append(txt)
        
        
        
        df = pd.DataFrame(data=special_data, columns=("pos", "row", "column", "value"))
        print("df:\n", df)
        
        lista = [df.columns[:,].values.astype(str).tolist()] + df.values.tolist()
        t1 = Table(lista)
        
        element.append(t1)
        return element
    
    
    
def create_output_pdf_summary(df, filebase):
    doc_summary  = SimpleDocTemplate("output/" + filebase + "_summary.pdf", pagesize=letter)
    
    element = []
    header = Paragraph("\nSummary of Analysis Run", styles["Heading1"])
    element.append(header)
    
    
    # sort by standard deviation  
    header = Paragraph("\nSorted by Interquartile Range", styles["Heading2"])
    element.append(header)
    
    df = df.sort_values(by=["interquartile_range"])
    lista = [df.columns[:,].values.astype(str).tolist()] + df.values.tolist()
    t1 = Table(lista)        
    element.append(t1)
    
    # sort by number of outliers  
    header = Paragraph("\nSorted by Number of Outliers", styles["Heading2"])
    element.append(header)
    
    df = df.sort_values(by=["# outliers"])
    lista = [df.columns[:,].values.astype(str).tolist()] + df.values.tolist()
    t1 = Table(lista)        
    element.append(t1)
    
    # sort by number of zeros  
    header = Paragraph("\nSorted by Number of Zeros", styles["Heading2"])
    element.append(header)
    
    df = df.sort_values(by=["# zeros"])
    lista = [df.columns[:,].values.astype(str).tolist()] + df.values.tolist()
    t1 = Table(lista)        
    element.append(t1)
    
    
    doc_summary.build(element)
    
    
    
def create_output_pdf(results, df_summary, filebase):    
    pdf_plots    = PdfPages("output/" +  filebase + "_plots.pdf")
    doc_outliers = SimpleDocTemplate("output/" + filebase + "_outliers.pdf", pagesize=letter)
    doc_zeros    = SimpleDocTemplate("output/" + filebase + "_zeros.pdf", pagesize=letter)
    
    element_outliers = []
    element_zeros = []
  

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
        plt.savefig(pdf_plots, format='pdf', bbox_inches='tight')
        plt.plot()
        #break
        
     
        # print outliers to PDF
        element_outliers = create_output_pdf_for_special_values(results[filename]["outliers_data"],\
                     element_outliers, filename)
        
        element_zeros = create_output_pdf_for_special_values(results[filename]["zeros_data"],\
                    element_zeros, filename)
    
    ## Doen with collecting data, create the documents
    doc_outliers.build(element_outliers)
    doc_zeros.build(element_zeros)
    

    pdf_plots.close()
 
 
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
    print("args: ", args)
    
    ### Get the configuration
    configuration = None
    input_file_full_path = os.path.join(os.getcwd(), args.input_config_file)
    with open(input_file_full_path, 'r') as stream:
        configuration = yaml.safe_load(stream)
        
    print(yaml.safe_dump(configuration, default_flow_style=False, default_style=None))
    
    read_file_or_dir = configuration["read_file_or_dir"]
    read_full_path = os.path.join(os.getcwd(), read_file_or_dir)
   
    
 
    df_summary = pd.DataFrame(columns=["filename",\
        "# outliers",\
        "# zeros",\
        "quartile_low",\
        "quartile_high",\
        "interquartile_range",\
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
                df_summary = open_file_and_calcualte(full_filename,\
                        results, df_summary, cutoff_offset_percent)
                
                
    else:
        df_summary = open_file_and_calcualte(read_full_path,\
                    results, df_summary, cutoff_offset_percent)
        
    
    from datetime import datetime
    now = datetime.now() # current date and time

    date_time = now.strftime("run_%m-%d-%Y--%H-%M")
    create_output_pdf(results, df_summary, date_time)
    create_output_pdf_summary(df_summary, date_time)
    
    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    
    
