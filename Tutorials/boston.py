# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:16:00 2023

@author: volka
"""
import mglearn
from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))

mglearn.plots.plot_knn_classification(n_neighbors=1)