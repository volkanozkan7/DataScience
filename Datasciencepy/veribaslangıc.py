# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri ön işleme
veriler = pd.read_csv("veriler.csv")
print(veriler)


boy = veriler[["boy"]]
print(boy)

boykilo = veriler[["boy","kilo"]]
print(boykilo)

class insan:
    boy=180
    def kosmak(self,b):
        return b+4
ali = insan()
print(ali.boy)
print(ali.kosmak(90))

#eksik veriler

eksikveriler = pd.read_csv("eksikveriler.csv")
print(eksikveriler)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = "mean")
Yas = veriler.iloc[:,1:4].values
print(Yas)
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)
#Yas column daki nan değerleri ortalama değere eşitledi
#fitle ortalama değer alıcagını öğretip transformla uyguluyoruz

