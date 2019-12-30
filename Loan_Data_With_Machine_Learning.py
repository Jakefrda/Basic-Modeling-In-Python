# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 08:12:06 2019

@author: JAKEFREDRICH
Machine Learning - Exploration

"""

# Import Packages required
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
#%matplotlib inline

''' IMPORT Loan Data'''

loan_df = pd.read_csv("loan_train.csv")

'''DATA VISUALIZATION AND ANALYSIS'''
loan_df.columns
print(loan_df.head())