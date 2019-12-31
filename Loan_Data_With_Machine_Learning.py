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


#!conda install -c anaconda seaborn -y
#!conda install -c anaconda pydotplus -y

'''******************************'''
''' IMPORT LOAN DATA '''
'''******************************'''


loan_df = pd.read_csv("loan_train.csv")


''' DATA VISUALIZATION AND ANALYSIS '''
'''
The following shows:
260 People have paid off their loan.
86 have gone into collection.
'''

loan_df.head()
print(loan_df.loan_status.value_counts())

loan_df['due_date'] = pd.to_datetime(loan_df['due_date']) # Convert Date column to Date Time Object
loan_df['effective_date'] = pd.to_datetime(loan_df['effective_date']) # Convert Date column to Date Time Object



'''*********************************************'''
'''  DATA VISUALIZATION - FEATURE EXPLORATION  '''
'''*********************************************'''


loan_df.columns
print(loan_df['loan_status'].value_counts())



#Plot Gender
print("Gender Values : " , loan_df.Gender.unique())
loan_df.Gender = pd.Categorical(loan_df.Gender) 
labels = 'male', 'female' #Create Labels 
sizes = [(loan_df.Gender == 'male').sum(),(loan_df.Gender == 'female').sum()] #Store sums of male/female
colors = ['lightcoral', 'lightskyblue']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()

#Plot education
print("Education Values : " , loan_df.education.unique())
loan_df.education = pd.Categorical(loan_df.education) 
labels = 'High School or Below', 'Bechalor', 'college', 'Master or Above'
sizes = [(loan_df.education == 'High School or Below').sum(),(loan_df.education == 'Bechalor').sum(), (loan_df.education == 'college').sum(), (loan_df.education == 'Master or Above').sum()] #Store sums of male/female
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
plt.pie(sizes, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.legend(labels, loc="best")
plt.axis('equal')
plt.show()

#Plot Principal
print("Principal Values : " , loan_df.Principal.unique())
loan_df.hist(column='Principal')

#Plot terms
print("Term Values : " , loan_df.terms.unique())
loan_df.hist(column='terms')

#Plot Age
print("Age Values : " , loan_df.age.unique())
loan_df.hist(column='age', bins=50)

# Remove unecessary columns
loan_df.columns
loan_df = loan_df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
print(loan_df.head())



'''************************************'''
'''  ADDITIONAL VISUALIZATION   '''
'''*************************************'''


import seaborn as sns

#Plot Loan Status by Principal for each Gender
bins = np.linspace(loan_df.Principal.min(), loan_df.Principal.max(), 10)
grid = sns.FacetGrid(loan_df, col="Gender", hue="loan_status", palette="Set2", col_wrap=2)
grid.map(plt.hist, "Principal", bins=bins, ec="k")

grid.axes[-1].legend()
plt.show()

#Plot Loan Status by Age for each Gender
bins = np.linspace(loan_df.age.min(), loan_df.age.max(), 10)
grid = sns.FacetGrid(loan_df, col="Gender", hue="loan_status", palette="Set2", col_wrap=2)
grid.map(plt.hist, 'age', bins=bins, ec="k")

grid.axes[-1].legend()
plt.show()