#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import time

https://www.datarobot.com/blog/ordinary-least-squares-in-python/

# data, this creates random data in three columns
# https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/

#https://www.statsmodels.org/dev/examples/notebooks/generated/ols.html


# load numpy and pandas for data manipulation
import numpy as np
import pandas as pd
# load statsmodels as alias ``sm``import statsmodels.api as sm
# load the longley dataset into a pandas data frame - first column (year) used as row labels
df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/longley.csv', index_col=0) 
df.head()

y = df.Employed  # response
X = df.GNP  # predictor
X = sm.add_constant(X)  # Adds a constant term to the predictor
X.head()

# build a dataset for the indendent variable


est = est.fit()
est.summary()




















# the scatter plot
# dependent variable
plt.scatter(df.index, df, s = 10, facecolors='none', edgecolors="brown")

# independent variable
plt.scatter(df_y.index, df_y, s = 10, facecolors='none', edgecolors="green")


# plot the fitted values from the model
plt.plot(df.index, model.fittedvalues, 'r--.', c="blue", label="OLS")
plt.plot()
  
