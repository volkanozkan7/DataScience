# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 23:39:37 2023

@author: volka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn 
dataframe = pd.read_csv("skoda.csv")

dataframe.head()
dataframe.describe()

dataframe.isnull().sum()

plt.figure(figsize = (7,5))
sbn.displot(dataframe["price"])
sbn.countplot(dataframe["year"])
dataframe.corr()


dataframe.corr()["price"].sort_values(ascending=False)
#fiyatı etkileyen en çok yıl değeri.

sbn.scatterplot(x = "mileage" , y="price",data=dataframe)
dataframe.sort_values("price",ascending=False).head(20)
len(dataframe)  #6267 değer var
#sadece tek bir tane pahalı araç var. Bunu datasetten atabiliriz

newdataframe = dataframe.sort_values("price",ascending=False).iloc[1:6267]
newdataframe.describe()
dataframe.describe()

plt.figure(figsize = (7,5))
sbn.distplot(newdataframe["price"])
dataframe.groupby("year").mean()["price"]

#sadece bir yılı çıkarmak istiyorsak.2004 ü çıkardık.
dataframe[dataframe.year !=2004].groupby("year").mean()["price"]

#transmission column atıyoruz gereksiz oldugu için.
dataframe = dataframe.drop("transmission",axis = 1)
dataframe = dataframe.drop("fuelType",axis = 1)
dataframe = dataframe.drop("model",axis = 1)
y = dataframe["price"].values
x = dataframe.drop("price",axis = 1).values

from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 42)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


X_train.shape

model = Sequential()
model.add(Dense(12,activation = "relu"))
model.add(Dense(6,activation = "relu"))
model.add(Dense(6,activation = "relu"))
model.add(Dense(6,activation = "relu"))
model.add(Dense(1,activation  = "sigmoid"))
model.compile(optimizer = "adam",loss= "mse")
model.fit(x = X_train,y = y_train, validation_data = (X_test,y_test),batch_size = 200, epochs = 300)

losedata = pd.DataFrame(model.history.history)
losedata.plot()
'''İki kayıp verisi hemen hemen aynı noktada ilerliyor. Eğer birisinde bir ayrım olsaydı
   bir hata oldugunu düşünebilirdik'''
   
from sklearn.metrics import mean_squared_error, mean_absolute_error

predict_frame = model.predict(X_test)
predict_frame
mean_absolute_error(y_test, predict_frame)
'''2047.4446031812784 birim absolute fark var. İstersek eğer epochs değeri yada diğer verileri değiştirip atabiliriz.
   Ancak bu değer 0 a çok yaklaşırsa overfitting olabilir.'''


plt.scatter(y_test, predict_frame)
plt.plot(y_test,y_test,"r-*") 


























