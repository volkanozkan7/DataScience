# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 19:10:08 2023

@author: volka
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_excel('Iris.xls')
#pd.read_csv("veriler.csv")
#test
print(veriler)

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.30, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred = logr.predict(X_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("LR")
print(cm)



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print("KNN")
print(cm)



from sklearn.svm import SVC
svc = SVC(kernel='sigmoid')
svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'gini',max_depth=100)

dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1, criterion = 'entropy')
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)


    
# 7. ROC , TPR, FPR değerleri 

y_proba = rfc.predict_proba(X_test)
print(y_test)
print(y_proba[:,0])

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr)
