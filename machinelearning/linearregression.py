# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:50:33 2023

@author: volka
"""
import mglearn
from  sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples = 60)
from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X, y, random_state = 42)

lr = LinearRegression().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))#The “slope” parameters (w), also called weights or coefficients, are stored in the coef_attribute
print("lr.intercept_: {}".format(lr.intercept_)) #while the offset or intercept (b) is stored in the intercept_ attribute

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

#'An R2 of around 0.66 is not very good, but we can see that the scores on the training
#and test sets are very close together. This means we are likely underfitting,not overfitting.
#For this one-dimensional dataset, there is little danger of overfitting, asthe
#modelis very simple (or restricted). However, with higher-dimensional datasets
#(meaning datasets with a large number of features), linear models become more powerful,
#and there is a higher chance of overfitting'