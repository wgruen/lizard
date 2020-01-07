#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt



# data, this creates random data in three columns
np.random.seed(123)
df = pd.DataFrame(np.random.randint(0,100,size=(100, 3)), columns=list('ABC'))


# assign dependent and independent / explanatory variables
variables = list(df.columns)
print("variables:\n", variables)
print("df:\n", df)

# y : column A will be the dependent variable
y = 'A'

# x : the other columns will be number of observations
x = [var for var in variables if var not in y ]
print("x\n", x)

# Ordinary least squares regression
print(df[y])
print(df[x])
model_Simple = sm.OLS(df[y], df[x]).fit()


# Add a constant term like so:
model = sm.OLS(df[y], sm.add_constant(df[x])).fit()
print("model.params:\n", model.params)

print(model.summary())

y = ['A', 'A']

#duplicate X as many times as needed
#y = np.repeat(df[y], len(x[0]))
print("len df[x]: ", len(df[x]))
print("len df[y]: ", len(df[y]))


#plt.subplot(1, 2, 1)
#plt.title(filename + " - " + str(len(results[filename]["Y"])) + " data poin
plt.scatter(df[y], df[x], s = 50, c="brown")
plt.plot()
  



