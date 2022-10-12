# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 19:11:17 2022

@author: Volkan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri ön işleme
veriler = pd.read_csv("veriler.csv")
print(veriler)

#eksik veriler
#veri yükleme



ulke = veriler.iloc[:,0:1].values
print(ulke)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()  #sayısal değer 1,2,3 gibi
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
print(ulke)
ohe = preprocessing.OneHotEncoder()  #column başlıklarına etiket taşıyıp 1 mi 0 mı değer vermek
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

c = veriler.iloc[:,-1:].values
print(c)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()  #sayısal değer 1,2,3 gibi
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(c)


ohe = preprocessing.OneHotEncoder()  #column başlıklarına etiket taşıyıp 1 mi 0 mı değer vermek
c = ohe.fit_transform(c).toarray()
print(c)

#verilerin birleştirilmesi
sonuc = pd.DataFrame(data=ulke,index = range(22),columns = ["fr","tr","us"])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas,index = range(22),columns=["boy","kilo","yas"])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data= c[:,:1],index= range(22),columns = ["cinsiyet"])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size= 0.33,random_state = 0)

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

boy  = s2.iloc[:,3:4].values
print(boy)
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis = 1)

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size= 0.33,random_state = 0)

regressor2 = LinearRegression()
regressor2.fit(x_train,y_train)

y_pred = regressor2.predict(x_test)

#verilere göre boyu tahmin  etti (y_predict de). Tam olarak düzgün tahmin edemedi ama yakın

import statsmodels.api as sm

X = np.append(arr = np.ones((22,1)).astype(int),values = veri,axis=1 )

X_l = veri.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()           #ols istatiksel verileri çıkartmaya yarıyor
print(model.summary())   #summaryde P>t değerinde 1 e yakın olan 5. özelliği atıcaz listeden 

X_l = veri.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit()           #ols istatiksel verileri çıkartmaya yarıyor
print(model.summary())
 









