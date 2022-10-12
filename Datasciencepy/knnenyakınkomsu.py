# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 12:34:07 2022

@author: Volkan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri ön işleme



#eksik veriler
#veri yükleme
veriler = pd.read_csv("veriler.csv")
print(veriler)

x = veriler.iloc[:,1:4].values   #Bağımsız değişkenler
y = veriler.iloc[:,4:].values   #bağımlı değişken


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size= 0.33,random_state = 0)

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state = 0)
logr.fit(X_train,y_train)   #train bilgisini öğrenmiştik

y_pred = logr.predict(X_test)  #şimdi tahmin edebiliriz
print(y_pred)
print(y_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10, metric= "minkowski")
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)