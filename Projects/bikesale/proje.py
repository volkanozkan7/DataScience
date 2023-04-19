# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 17:55:34 2023
@author: volka
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import mglearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel("proje.xlsx",index_col = False,names= ["ID","Marital Status","Gender", "Income","Children", "Education", "Occupation","Home Owner",
"Cars","Commute Distance","Region","Age","Age Brackets","Purchased Bike"])
data = data[["Marital Status","Gender","Income","Cars", "Region","Age","Purchased Bike"]]

x = data.iloc[:,1:6].values #bağımsız değişkenler
y = data.iloc[:,6:].values #bağımlı değişken
marital = data.iloc[:,0:1].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
marital[:,0] = le.fit_transform(data.iloc[:,0])

ohe = preprocessing.OneHotEncoder()
ohe = ohe.fit_transform(marital).toarray()

gender = data.iloc[:,1:2].values
ge = preprocessing.LabelEncoder()
geh = preprocessing.OneHotEncoder()
geh = geh.fit_transform(gender).toarray()

region = data.iloc[:,4:5].values
re = preprocessing.LabelEncoder()
reh = preprocessing.OneHotEncoder()
reh = reh.fit_transform(region).toarray()

bisiklet = data.iloc[:,-1].values
ıncome = data.iloc[:,2:3].values
cars = data.iloc[:,3:4].values
age = data.iloc[:,5:6].values


sonuc = pd.DataFrame(data = ohe, index = range(1000), columns = ["Married","Single"])
sonuc2 = pd.DataFrame(data = geh, index = range(1000), columns = ["Female","Male"])
sonuc3 = pd.DataFrame(data = reh, index = range(1000), columns = ["Europe","North America","Pacific"])
sonuc4= pd.DataFrame(data = ıncome, index = range(1000), columns = ["Income"])
sonuc5 = pd.DataFrame(data = cars, index = range(1000), columns = ["Cars"])
sonuc6 = pd.DataFrame(data = age, index = range(1000), columns = ["Age"])
sonuc7 = pd.DataFrame(data = bisiklet, index = range(1000), columns = ["Purchased Bike"])

s = pd.concat([sonuc,sonuc2], axis = 1)
s2 = pd.concat([s,sonuc3], axis= 1)
s3 = pd.concat([s2,sonuc4], axis= 1)
s4 = pd.concat([s3,sonuc5], axis= 1)
s5 = pd.concat([s4,sonuc6], axis= 1)
s6 = pd.concat([s5,sonuc7], axis= 1)

x_train, x_test,y_train,y_test = train_test_split(s5,sonuc7,test_size=0.33, random_state=0)

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)
cm = confusion_matrix(y_test,y_pred)
print('LR')
print(cm)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=500, criterion = 'entropy')
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("Knn")
print(cm)

from sklearn.svm import SVC

svm = SVC(C=100, kernel = "rbf")
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))
cm = confusion_matrix(y_test,y_pred)
print("SVC")
print(cm)





















