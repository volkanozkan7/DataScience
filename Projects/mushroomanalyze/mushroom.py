# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:42:40 2023

@author: volka
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression , RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

mush = pd.read_csv("mushrooms.csv")
mush.head()
classes = mush["class"].value_counts()
classes
plt.bar("Edible", classes["e"])
plt.bar("Poisonous", classes["p"])
plt.show()

x = mush.loc[:, ["cap-shape","cap-color", "ring-number","ring-type"]]
y = mush.loc[:, "class"]

encode = LabelEncoder()
for i in x.columns:
    x[i] = encode.fit_transform(x[i])
    
y = encode.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)
lr = LogisticRegression()
rc = RidgeClassifier()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()    
nb = GaussianNB()
neural = MLPClassifier()

lr.fit(X_train,y_train)
rc.fit(X_train,y_train)
dt.fit(X_train,y_train)
rf.fit(X_train, y_train)
nb.fit(X_train,y_train)
neural.fit(X_train,y_train)

lr_pred = lr.predict(X_test)
ridge_pred = rc.predict(X_test)
decision_tree_pred = dt.predict(X_test)
random_forest_pred = rf.predict(X_test)
nb_pred = nb.predict(X_test)
neural_pred = neural.predict(X_test)

lr_report = classification_report(y_test, lr_pred)
ridge_report = classification_report(y_test, ridge_pred)
decision_report = classification_report(y_test, decision_tree_pred)
random_report = classification_report(y_test, random_forest_pred)
nb_report = classification_report(y_test, nb_pred)
neural_report = classification_report(y_test, neural_pred)

print("     Logistic Regression     ")
print(lr_report)

print("     Ridge Regression     ")
print(ridge_report)

print("     Decision Tree     ")
print(decision_report)

print("     Random Forest    ")
print(random_report)

print("     Naive Bayes     ")
print(nb_report)

print("     Neural Network     ")
print(neural_report)






















