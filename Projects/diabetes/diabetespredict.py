# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:12:08 2023

@author: volka
"""

import numpy as np
import pandas as pd

dia = pd.read_csv("diabetes.csv")
dia.head()
dia.info()
dia.isnull().sum()

dia.eq(0).sum()
dia.shape
#768 hasta değeriöiz var.500 ü 0 otucome yani hasta değil. 8 girdi özellik bir çıktı özelliğimiz var
dia[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]= dia[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]].replace(0,np.NaN)
#0 Değerler yerine nan değerler verdik. Şimdi bunlara değer vericez. Mean değerler ile doldurucaz.

dia.fillna(dia.mean(),inplace = True)
dia.head()
dia.isnull().sum()
dia.eq(0).sum()

#correlation
dia.corr()

import seaborn as sns
sns.heatmap(dia.corr())

#Korelasyona bakarak en yüksek korelasyonlu özellikleri inceleyeceğiz.
feature_names = dia.corr().nlargest(4,"Outcome").index.tolist()
feature_names

from sklearn import linear_model
from sklearn.model_selection import cross_val_score

x = dia[["Glucose","BMI","Age"]]
y = dia[["Outcome"]]

lr = linear_model.LogisticRegression()
lr_score = cross_val_score(lr,x,y,cv= 10,scoring = "accuracy").mean()

results = []
results.append(lr_score)
from sklearn import svm
lr_svm = svm.SVC(kernel = "linear")
lr_svm_score = cross_val_score(lr_svm,x,y ,cv = 10 ,scoring = "accuracy").mean()

results.append(lr_svm_score)

import pickle
filename = "diabets.sav"
lr.fit(x,y)
pickle.dump(lr,open(filename,"wb"))  #write binary, yazma amaçlı
load_model = pickle.load(open(filename,"rb"))  #read binary okuma amaçlı

#1 = diabet , 0 = diabet değil

Resultsdia = []
Glucose = 60
BMI = 53
Age = 46
Resultsdia.append(Glucose)

predict = load_model.predict([[Glucose,BMI,Age]])














