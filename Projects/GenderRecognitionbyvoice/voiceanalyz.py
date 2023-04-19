# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:53:29 2023

@author: volka
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("voice.csv")
print(data.head())
print(data.corr())

print(data.isnull().sum())

data.shape
data["label"].hist()

x = data.iloc[:, :-1]
y = data.iloc[:,-1]
from sklearn.preprocessing import LabelEncoder
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
y
data["y"] = y
data = data.drop('label', axis=1)
data.rename(columns = {'y' : 'Gender'}, inplace = True)

#Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
X = scaler.transform(x)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.svm import SVC

print("Size of training set: {} size of test set: {}".format(X_train.shape[0], X_test.shape[0]))
best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
         # for each combination of parameters, train an SVC
         svm = SVC(gamma=gamma, C=C)
         svm.fit(X_train, y_train)
         #evaluate the SVC on the test set
         score = svm.score(X_test, y_test)
         # if we got a better score, store the score and parameters
         if score > best_score:
            best_score = score
            best_parameters = {'C': C, 'gamma': gamma}
            
print("Best score: {:.2f}".format(best_score))
print("Best parameters: {}".format(best_parameters))

#GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

from sklearn.model_selection import cross_val_score
svc=SVC(kernel='rbf')
scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') 
print(scores.mean())

C_range=list(range(1,20))
acc_score=[]
for c in C_range:
    svc = SVC(kernel='rbf', C=c)
    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')
    acc_score.append(scores.mean())
print(acc_score)    


C_values=list(range(1,20))
plt.plot(C_values,acc_score)
plt.xticks(np.arange(0,21,2))
plt.xlabel('Value of C for SVC')
plt.ylabel('Cross-Validated Accuracy')

#heatmap
corr_matrix = data.corr().abs()
threshold = 0.5
filtered_corr_df = corr_matrix[(corr_matrix >= threshold) & (corr_matrix != 1.000)] 
plt.figure(figsize=(30,10))
sns.heatmap(filtered_corr_df, annot=True, cmap="Reds")
plt.show()

plt.figure(figsize=(12,12))
high_corr_columns = data.columns[list(data.corr().apply( lambda value: value >0.5 ).sum()>1)]
corr = data.corr().loc[high_corr_columns,high_corr_columns]
sns.heatmap(corr,square=True,center=0,annot=True)




