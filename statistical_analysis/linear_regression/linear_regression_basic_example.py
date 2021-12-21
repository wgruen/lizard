#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import time


# data, this creates random data in three columns
# https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/

#https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html


np.random.seed(int(time.time()))
df1__0_25 = pd.DataFrame(np.random.randint(0, 25, size=(25, 1)), columns=list('A'))
df2_26_50 = pd.DataFrame(np.random.randint(26, 50,size=(25, 1)), columns=list('A'))
df3_51_75 = pd.DataFrame(np.random.randint(51, 75,size=(25, 1)), columns=list('A'))
df4_76_100 = pd.DataFrame(np.random.randint(76, 100,size=(25, 1)), columns=list('A'))
#print("df2:\n", df2)



# build a dataset for the indendent variable
df = pd.DataFrame()
df = df.append(df1__0_25, ignore_index=True)
df = df.append(df2_26_50, ignore_index=True)
df = df.append(df3_51_75, ignore_index=True)
df = df.append(df4_76_100, ignore_index=True)

X = df
# assign dependent and independent / explanatory variables
#variables = list(df.columns)
#print("variables:\n", variables)
#print("df:\n", df)

# build a dataset for the dependent variable
df_y = pd.DataFrame()
df_y = df_y.append(df2_26_50, ignore_index=True)
df_y = df_y.append(df2_26_50, ignore_index=True)
df_y = df_y.append(df2_26_50, ignore_index=True)
df_y = df_y.append(df2_26_50, ignore_index=True)

Y = df_y
print("Y the dependent variable:\n",Y)



# column B and C will be the independent variable
#x = columns=list('B')

# adding a constant to the two variables
X = sm.add_constant(df)
print("X two variables:\n", X)



# Add a constant term like so:
model = sm.OLS(Y, X).fit()
print("model.params:\n", model.params)   
print(model.summary())


# the scatter plot
# dependent variable
plt.scatter(df.index, df, s = 10, facecolors='none', edgecolors="brown")

# independent variable
plt.scatter(df_y.index, df_y, s = 10, facecolors='none', edgecolors="green")


# plot the fitted values from the model
plt.plot(df.index, model.fittedvalues, 'r--.', c="blue", label="OLS")
plt.plot()
  
