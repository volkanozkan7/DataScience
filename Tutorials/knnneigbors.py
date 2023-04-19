# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 15:01:09 2023

@author: volka
"""

import pandas as pd
import numpy as np
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

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
iris_dataset['target_names'][prediction]))


y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))