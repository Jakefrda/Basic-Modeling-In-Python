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
#!conda install -c conda-forge pydotplus -y


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



#Plot Day of Week
loan_df['dayofweek'] = loan_df['effective_date'].dt.dayofweek
bins = np.linspace(loan_df.dayofweek.min(), loan_df.dayofweek.max(), 10)
grid = sns.FacetGrid(loan_df, col="Gender", hue = "loan_status", palette="Set2", col_wrap=2)
grid.map(plt.hist, "dayofweek", bins=bins, ec="k")
grid.axes[-1].legend()
plt.show()
#Interesting Note: It seems that a majority of the collection loans were taken out at the end of the week.
#Using Feature binarization we can create a Weekend threshold value (Monday:Thursday = 0 ; Friday:Sunday = 1)¶


'''************************************'''
''' PRE-PROCESSING: FEATURE SELECTION   '''
'''*************************************'''

# identify weekend
loan_df['weekend'] = loan_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
loan_df.head()

# Convert Categorical features to numerical values
loan_df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)



# Convert male and female to 0 and 1
loan_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
loan_df.head()

# One Hot Encoding - use one hot encoding to convert categorical education field to binary fields
loan_df.groupby(['education'])['loan_status'].value_counts(normalize=True)


Features = loan_df[['Principal','terms','age','Gender','weekend']]
Features = pd.concat([Features,pd.get_dummies(loan_df['education'])], axis=1)
Features.drop(['Master or Above'], axis = 1,inplace=True) # Percentage of population is too small
Features.head()

# Feature Sets: X
X = Features
X[0:5]



# Label: y
y = loan_df['loan_status'].values
y[0:5]

# Normalize Data - Equilize the range and data variabliity.  This reduces bias from feature size difference
X= preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]



'''************************************'''
''' CLASSIFICATION MODELING   '''
'''*************************************'''
# K Nearest Neighbor (KNN)
# Decision Tree
# Support Vector Machine (SVM)
# Logistic Regression



print(loan_df.dtypes)

'''************************************'''
''' KNN - K-Nearest Neighbors  '''
'''*************************************'''

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn import tree

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4) # Split data into Testing and Training Data.  80% Testing, 20% Training
print ('Train set:', X_train.shape,  y_train.shape) # Confirm shape
print ('Test set:', X_test.shape,  y_test.shape) # Confirm shap

k = 4 # Start with 4, will check for best k
kNN_cls = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train) # Fit the KNN Model using training data
kNN_cls

yhat = kNN_cls.predict(X_test)# Predict using Test Data Features


Ks = 25 # Number of K's to check
mean_acc = np.zeros((Ks-1)) # Create array of ks - 1 filled with zeros.  This will be populated with mean_accuracy in the loop below.
std_acc = np.zeros((Ks-1)) # Create array of ks - 1 filled with zeros.  This will be populated with standard_accuracy in the loop below.
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict values
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train) # Fit Model with n
    yhat=neigh.predict(X_test) # predict yhat using X_test
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat) # store accuracy score in mean_acc array

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0]) # store std_acc in array

#mean_acc
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)



#Display how our different K values did in accuracy
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

## Final Model uses k=7
k = 7
#Train Model and Predict  
kNN_cls = KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
kNN_cls



yhat_knn = kNN_cls.predict(X_test)# Predict using Test Data 
perckNN = metrics.accuracy_score(y_test, yhat_knn) # store accuracy score in mean_acc array
print("KNN Accuracy percentage" , perckNN)

JaccardkNN = jaccard_similarity_score(y_test, yhat_knn)
print("KNN Jaccard index: %.2f" % JaccardkNN)

F1ScorekNN = f1_score(y_test, yhat_knn, average='weighted')
print("KNN F1-score: %.2f" % F1ScorekNN )


'''************************************'''
''' Decision Tree  '''
'''*************************************'''


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics



loanTree = DecisionTreeClassifier(criterion="entropy", max_depth = 5) # Classify Decision Tree characteristics
loanTree.fit(X_train,y_train) # Fit Decision Tree using training set

predTree = loanTree.predict(X_test) # predict loan outcome using test set
print("DT's Accuracy: ", metrics.accuracy_score(y_test, predTree))
JaccardDT = jaccard_similarity_score(y_test, predTree)
print("DT's Jaccard index: %.2f" % JaccardDT)
F1ScoreDT = f1_score(y_test, predTree, average='weighted')
print("DT's F1-score: %.2f" % F1ScoreDT  )

predTree

# Check which Depth has the best accuracy


k=10 #Check 1 through 10 depth options
loanTree_acc=[]

for x in range(1,k):
    loanTree1=DecisionTreeClassifier(criterion='entropy', max_depth=x).fit(X_train,y_train.ravel()) # Fit Decision Tree using x depth on training data
    predTree1=loanTree1.predict(X_test) # Predict using test data
    loanTree_acc.append(metrics.accuracy_score(y_test, predTree1)) # Store Accuracy in list

loanTree_acc=np.asarray(loanTree_acc) # Convert Accuracy List to Array
print("DecisionTrees's Accuracy highest accuracy is ", loanTree_acc.max(), 'at max_depth: ', loanTree_acc.argmax()+1)

# Visualize the Decision Tree

from sklearn.externals.six import StringIO
#import pydotplus
import matplotlib.image as mpimg
from sklearn import tree
#%matplotlib inline 
#conda install graphviz


###Requires Pydotplus to run###
#dot_data = StringIO()
#filename = "drugtree.png"
#featureNames = Features.columns[0:9]
#targetNames = loan_df["loan_status"].unique().tolist()
#out=tree.export_graphviz(loanTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png(filename)
#img = mpimg.imread(filename)
#plt.figure(figsize=(100, 200))
#plt.imshow(img,interpolation='nearest')
#plt.show()


'''************************************'''
''' (SVM) Support Vector Machine  '''
'''*************************************'''


from sklearn import svm
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC

SVM_cls = svm.SVC(kernel='rbf') 
SVM_cls.fit(X_train, y_train) # fit SVM model with training data



SVM_yhat = SVM_cls.predict(X_test) # Predict using Test Data
SVM_yhat [0:5]

print("SVMs's Accuracy: ", metrics.accuracy_score(y_test, SVM_yhat))
JaccardSVM = jaccard_similarity_score(y_test, SVM_yhat)
print("SVM's Jaccard index: %.2f" % JaccardSVM)
F1ScoreSVM = f1_score(y_test, SVM_yhat, average='weighted')
print("SVM's F1-score: %.2f" % F1ScoreSVM  );



kernel ={'linear', 'rbf','poly'} # Check test accuracy using different kernel settings


for index,value in enumerate(kernel):
    SVM_cls=SVC(kernel=value).fit(X_train, y_train.ravel())
    s=SVM_cls.predict(X_test)
    print('accuracy of %s kernel is' %value , metrics.accuracy_score(y_test,SVM_yhat))



'''************************************'''
''' Logistic Regression  '''
'''*************************************'''


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

LR_cls = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train) # Fit Logistic Regression Model using Training Data
LR_cls



LR_yhat = LR_cls.predict(X_test) # Predict using Test Data
print(LR_yhat )



yhat_prob = LR_cls.predict_proba(X_test)
logLR = log_loss(y_test, yhat_prob)
print ("LogLoss: : %.2f" % logLR)
print("LRs's Accuracy: ", metrics.accuracy_score(y_test, LR_yhat))

JaccardLR = jaccard_similarity_score(y_test, LR_yhat)
print("LR's Jaccard index: %.2f" % JaccardLR)
F1ScoreLR = f1_score(y_test, LR_yhat, average='weighted')
print("LR's F1-score: %.2f" % F1ScoreLR)



'''************************************'''
''' REPORT  '''
'''*************************************'''

data_report =np.array([['Algorithm','Jaccard','F1-score','LogLoss'],['KNN',JaccardkNN ,F1ScorekNN,'NA'],['Decision Tree',JaccardDT,F1ScoreDT,'NA'], ['SVM',JaccardSVM,F1ScoreSVM,'NA'],
     ['LogisticRegression',F1ScoreLR,JaccardLR, logLR]])


print(pd.DataFrame(data = data_report[1:,1:], index = data_report[1:,0],columns = data_report[0,1:]))
