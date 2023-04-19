# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 22:36:21 2023

@author: volka
"""

import pandas as pd

url = "http://bilkav.com/satislar.csv"
veriler = pd.read_csv(url)
veriler = veriler.values

X = veriler[:,0:1]
Y = veriler[:,1]

bolme = 0.33


from sklearn import model_selection
X_train, X_test , Y_train, Y_test = model_selection.train_test_split (X,Y, test_size = bolme)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,Y_train)
print(lr.predict(X_test))

import pickle

dosya = "model.kayit"

pickle.dump(lr,open(dosya,"wb"))

yuklenen = pickle.load(open(dosya,"rb"))
print(yuklenen.predict(X_test))