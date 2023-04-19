# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:51:39 2023

@author: volka
"""

import numpy as np
x = np.array([[1,2,3],[4,5,6]])
print("x:\n{}".format(x))

import matplotlib.pyplot as plt
x = np.linspace(-10,10,100)
y = np.sin(x)
plt.plot(x,y,marker="*")

import pandas as pd
data = {'Name': ["John", "Anna", "Peter", "Linda"],
'Location' : ["New York", "Paris", "Berlin", "London"],
'Age' : [24, 13, 53, 33]
}
data_pandas = pd.DataFrame(data)

display(data_pandas)

display(data_pandas[data_pandas.Age > 30])