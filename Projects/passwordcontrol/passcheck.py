# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 15:51:52 2023

@author: volka
"""

import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data.csv",error_bad_lines=False)
data
data.isnull().sum()
data.dropna(inplace = True)
password_tuple = np.array(data)
password_tuple

data["strength"] = data["strength"].map({0: "Weak",1: "Medium",2: "Strong"})
sns.set_style('whitegrid')
sns.countplot(x = 'strength' , data = data)

x = np.array(data["password"])
y = np.array(data["strength"])

def word(password):
    character=[]
    for i in password:
        character.append(i)
    return character

tfid = TfidfVectorizer(tokenizer = word)
x = tfid.fit_transform(x)
import xgboost as xgb
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski')
knn.fit(X_train,y_train)
knn.score(X_test,y_test)

