# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:37:10 2023

@author: volka
"""
import mglearn
import pandas as pd
from sklearn.datasets import load_iris
iris_dataset = load_iris()


print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))

print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris_dataset["data"],iris_dataset["target"],random_state = 0)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

grr = pd.scatter_matrix(iris_dataframe, c=y_train,figsize=(15,15),marker="o",hist_kwds={"bins": 20},s = 60,alpha=.8,cmap=mglearn.cm3)