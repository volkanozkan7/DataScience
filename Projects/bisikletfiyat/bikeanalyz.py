# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 20:56:11 2023

@author: volka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

bike = pd.read_excel("bisiklet_fiyatlari.xlsx")

bike.head()
bike["BisikletOzellik1"]
bike[:20].plot(kind  = "barh")
plt.show()
sbn.pairplot(bike)
bike.describe()

from sklearn.model_selection import train_test_split
y = bike["Fiyat"].values
x = bike[["BisikletOzellik1","BisikletOzellik2"]].values

X_train,X_test, y_train, y_test = train_test_split(x,y,test_size = 0.33, random_state = 42)
X_train.shape
X_test.shape

#scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Dense(5, activation  ="relu"))
model.add(Dense(5, activation  ="relu"))
model.add(Dense(5, activation  ="relu"))
model.add(Dense(1))
model.compile(optimizer = "adam", loss = "mse")  #adam yerine rmsprop. mse mean squarred error deniyor.

#Çok büyük bir veri seti varsa eğer batch uygulanabilir. Verileri böler.

model.fit(X_train,y_train,epochs = 250)
loss = model.history.history["loss"]
sbn.lineplot(x=range(len(loss)),y  = loss)

train_loss = model.evaluate(X_train,y_train,verbose = 0)
test_loss =model.evaluate(X_test,y_test,verbose = 0)
train_loss
test_loss
#Bu değerlerin birbirine yakınlıgı önemli

test_predict = model.predict(X_test)
predict_df = pd.DataFrame(y_test,columns=["Real Y Values"])
test_predict = pd.Series(test_predict.reshape(330,))   #x_Test 330 shape di
predict_df = pd.concat([predict_df,test_predict],axis = 1)  #birleştirme işlemi
predict_df
predict_df.columns = ["Gerçek Y","Tahmin Y"]
predict_df

sbn.scatterplot(x = "Gerçek Y", y = "Tahmin Y", data = predict_df)

new_bike = [[1760,1758]]
new_bike = scaler.transform(new_bike)
model.predict(new_bike)





























