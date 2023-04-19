# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 12:20:49 2023

@author: volka
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import mglearn
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
   clf = model.fit(X, y)
   mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
   ax=ax, alpha=.7)
   mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
   ax.set_title("{}".format(clf.__class__.__name__))
   ax.set_xlabel("Feature 0")
   ax.set_ylabel("Feature 1")
axes[0].legend()