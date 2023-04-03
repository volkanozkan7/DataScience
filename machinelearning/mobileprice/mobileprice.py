# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 12:30:58 2023

@author: volka
"""

'''Price Range;
0 = 0 - 1500
1 = 1500 - 4000
2 = 4000 - 8000
3 = 8000 - 15000
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mobile = pd.read_csv("mobileprice.csv")
mobile.head()
mobile.info()
mobile.describe()

sns.jointplot(x ="ram",y = "price_range",data = mobile, color = "blue",kind = "reg")
sns.pointplot(y = "battery_power",x = "price_range", data = mobile)
sns.pointplot(y = "int_memory", x = "price_range",data = mobile)

labels = ["Has touchscreen","No touchscreen"]
values = mobile["touch_screen"].value_counts().values


fig1, (ax1,ax2) = plt.subplots(2)
ax1.pie(values,labels = labels,shadow = True,autopct="%1.1f%%",startangle = 90)
labels4g = ["4G supported",'Not supported']
values4g = mobile['four_g'].value_counts().values
ax2.pie(values4g, labels=labels4g, autopct='%1.1f%%',shadow=True,startangle=90)
plt.show()

sns.boxplot(x = "price_range", y = "battery_power", data = mobile)
sns.boxplot(x = "price_range", y = "ram", data = mobile)


plt.figure(figsize=(10,6))
mobile['fc'].hist(alpha=0.5,color='blue',label='Front camera')
mobile['pc'].hist(alpha=0.5,color='red',label='Primary camera')
plt.legend()
plt.xlabel('MegaPixels')

fig1, ax = plt.subplots()
wifi = ["Wifi","No Wifi"]
valueswifi = mobile['wifi'].value_counts().values
ax.pie(valueswifi,labels = wifi ,shadow = True,autopct="%1.1f%%",startangle = 90)


y = mobile['price_range']
X = mobile.drop('price_range',axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)
logr.score(X_test,y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10, metric='minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("KNeighbors")
print(cm)
knn.score(X_test,y_test)

from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)
svc.score(X_test,y_test)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)
gnb.score(X_test,y_test)

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy')
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)
dtc.score(X_test,y_test)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200, criterion = 'entropy')   
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)
rfc.score(X_test,y_test)





















