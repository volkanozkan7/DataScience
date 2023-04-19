# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:07:59 2023

@author: volka
"""

from sklearn.model_selection import train_test_split
import mglearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

data = pd.read_csv("adult-all.csv",index_col = False,names= ["age","workclass", 'fnlwgt', 'education', 'education-num',
'marital-status', 'occupation', 'relationship', 'race', 'gender',
'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
'income'])

data = data[["age","workclass","education","gender","hours-per-week","occupation","income"]]

print(data.head())

#datada hatalı değerler yanlış sınıflandırılmalar var mı ona bakılır
print(data.gender.value_counts())  #male ve female var. Sorun yok

print("Original Features:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("Features after get_dummies:\n", list(data_dummies.columns))

display(data_dummies.head())

features = data_dummies.loc[:, 'age':'occupation_Transport-moving']
# Extract NumPy arrays
X = features.values
y = data_dummies['income_>50K'].values
print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))