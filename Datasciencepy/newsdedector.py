# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 16:10:37 2023

@author: volka
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

nd = pd.read_csv("news.csv")
print(nd.shape)

labels = nd.label
print(labels.head())

x_train,x_test,y_train,y_test = train_test_split(nd["text"], labels, test_size= 0.3, random_state= 7)
tfid_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfid_train=tfid_vectorizer.fit_transform(x_train) 
tfid_test=tfid_vectorizer.transform(x_test)

from sklearn.metrics import confusion_matrix


#passiveagressive
pac=PassiveAggressiveClassifier(max_iter=500)  #max iter 1000
pac.fit(tfid_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfid_test)
pascore=accuracy_score(y_test,y_pred)
print(f"Passive agressive test result : {round(pascore*100,2)}%")
cm = confusion_matrix(y_test,y_pred)
print(cm)

#decision Tree
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(tfid_train,y_train)
y_pred = dtc.predict(tfid_test)
pascore=accuracy_score(y_test,y_pred)
print(f"Decision Tree test result : {round(pascore*100,2)}%")
cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)


#KNN
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(tfid_train,y_train)
y_pred = knn.predict(tfid_test)
knnscore=accuracy_score(y_test,y_pred)
print(f"KNN test result : {round(knnscore*100,2)}%")
cm = confusion_matrix(y_test,y_pred)
print(cm)

#SVC
svc = SVC(kernel='rbf',C=1000)
svc.fit(tfid_train,y_train)
y_pred = svc.predict(tfid_test)
svcscore=accuracy_score(y_test,y_pred)
print(f"SVC test result : {round(svcscore*100,2)}%")
cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)








