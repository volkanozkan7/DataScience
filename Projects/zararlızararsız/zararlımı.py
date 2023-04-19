# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 18:25:09 2023

@author: volka
"""

import pandas as pd
import numpy as np
dataFrame = pd.read_excel("maliciousornot.xlsx")
dataFrame
dataFrame.info()
dataFrame.describe()
dataFrame.corr()["Type"].sort_values()

import matplotlib.pyplot as plt
import seaborn as sbn

sbn.countplot(x="Type", data = dataFrame)
dataFrame.corr()["Type"].sort_values().plot(kind="bar")
y = dataFrame["Type"].values
x = dataFrame.drop("Type",axis = 1).values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=15)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
x_train.shape

model = Sequential()

model.add(Dense(units=30,activation = "relu"))
model.add(Dense(units=15,activation = "relu"))
model.add(Dense(units=15,activation = "relu"))
model.add(Dense(units=1,activation = "sigmoid"))

model.compile(loss="binary_crossentropy",optimizer = "adam")
model.fit(x=x_train, y=y_train, epochs=700,validation_data=(x_test,y_test),verbose=1)

model.history.history

modelKaybi = pd.DataFrame(model.history.history)
modelKaybi.plot()

'''Loss ve valudationloss arasındaki fark epochs değerinin fazla olması ve overfittinge yol açması oldu. Bu durumda
   earlystopping fonksiyonu kullanılarak bu değerlerin ayrılmaya başladıgı anda eğitimi durdurabilmeyi sağlıyoruz.'''
   
model = Sequential()

model.add(Dense(units=30,activation = "relu"))
model.add(Dense(units=15,activation = "relu"))
model.add(Dense(units=15,activation = "relu"))
model.add(Dense(units=1,activation = "sigmoid"))

model.compile(loss="binary_crossentropy",optimizer = "adam")
earlyStopping = EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)
model.fit(x=x_train, y=y_train, epochs = 700, validation_data = (x_test,y_test), verbose = 1, callbacks=[earlyStopping])
'''Eğitim sırasında 57/700 de durdu. Demekki burdan sonra değerler arasında ayrım başlıyordu'''

modelKaybi = pd.DataFrame(model.history.history)
modelKaybi.plot()

model = Sequential()

model.add(Dense(units=30,activation = "relu"))
model.add(Dropout(0.6))

model.add(Dense(units=15,activation = "relu"))
model.add(Dropout(0.6))

model.add(Dense(units=15,activation = "relu"))
model.add(Dropout(0.6))

model.add(Dense(units=1,activation = "sigmoid"))

model.compile(loss="binary_crossentropy",optimizer = "adam")
model.fit(x=x_train, y=y_train, epochs = 700, validation_data = (x_test,y_test), verbose = 1, callbacks=[earlyStopping])

kayipDf = pd.DataFrame(model.history.history)
kayipDf.plot()
predict_x=model.predict(x_test) 


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predict_x))
print(confusion_matrix(y_test,predict_x.round()))





















