"""
1.  provide file
2.  encode output
3.  split train and test values from given file, selecting columns for data and results
4.  fit through scaler for train and test
5.  use a multilayer perceptron to fit training values
6.  run predictions for test values
7.  evaluate with confusion matrix

fields with * can be produced outside

incoming data:
*- file with the source

outgoing/incoming data:
*- column labels (decision to process columns)
    *- decide columns to evaluate
    *- decide the result table for training purposes

outgoing only fields:
- predictions
- confusion matrix
- classification report
"""

#import numpy as np
from multiprocessing.connection import answer_challenge
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

# 1
df = pd.read_csv("pred.csv")

#print("data: \n", df)
#print(df.head(n=3))

#------------------------------------------------------------------------------------------------------------------------------------

# 2
le = preprocessing.LabelEncoder()

labelList       =   ['A','B','C','D']
answ            =   'E'
toPredict       =   [[12.3,33,250,True]]

# result = le.fit_transform(df['E'])

#------------------------------------------------------------------------------------------------------------------------------------
##### Desde 3

L = pd.DataFrame()
firstRun = True
for x in labelList:
    pivot = pd.DataFrame(df[x])
    if (firstRun):
        L = pivot
        firstRun = False
    else:
        L = pd.concat([L, pivot.reindex(L.index)], axis=1)

L_answ = pd.DataFrame(df[answ])
x_train, x_test, y_train, y_test = train_test_split(L, L_answ, test_size=0.5, random_state=13)

#------------------------------------------------------------------------------------------------------------------------------------
### 4

scaler  =   StandardScaler()
scaler.fit(x_train)

x_train =   scaler.transform(x_train)
x_test  =   scaler.transform(x_test)

#------------------------------------------------------------------------------------------------------------------------------------
### 5

MLP     =   MLPClassifier(hidden_layer_sizes=(13,13,13), max_iter=1000)
MLP.fit(x_train, y_train.values.ravel())

#------------------------------------------------------------------------------------------------------------------------------------
### 6

predictions = MLP.predict(x_test)
print("Predictions: \n",predictions)

prediction = MLP.predict(toPredict)
print("Prediction: ", prediction)

#------------------------------------------------------------------------------------------------------------------------------------
### 7

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
