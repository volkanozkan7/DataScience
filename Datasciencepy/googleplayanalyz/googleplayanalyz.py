# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:06:37 2023

@author: volka
"""

import pandas  as pd
import seaborn as sns

gp = pd.read_csv("googleplaystore.csv")
print(gp.head())

print(gp.columns) #verilerin arasında boşluklar var
gp.columns = gp.columns.str.replace(" ","_")  #boşluk yerine _ koyduk
print(gp.columns)

print(gp.shape)
print(gp.dtypes)  #Size,rewiews, installs gibi veriler object tipinde. sayısala çevirmemiz gerekiyor

#eksik veriler
print(gp.isnull().sum())
sns.set_theme()
sns.set(rc={"figure.dpi":300,"figure.figsize":(12,9)})
sns.heatmap(gp.isnull(), cbar = False)
rating_median = gp["Rating"].median()
print(rating_median)

gp["Rating"].fillna(rating_median,inplace = True)
gp.dropna(inplace=True)
print(gp.isnull().sum())


gp["Reviews"] = gp["Reviews"].astype("int64")  #reviews integera çevirdik
print(gp["Reviews"].describe().round())


print(gp["Size"].unique())  #Size verilerine tek değerler var.

gp["Size"].replace("M","",regex = True, inplace  = True)
gp["Size"].replace("k","",regex = True, inplace  = True)  #m ve k harfleri kaldırıldı

print(gp["Size"].unique())

#size kümesinde yabancı bir veri var ondalık tipe çevirip medianı aldık
size_median = gp[gp["Size"]!= "Varies with device"]["Size"].astype(float).median()
print(size_median)

#metin yerine median değeri atama
gp["Size"].replace("Varies with device",size_median,inplace= True)
gp.Size = pd.to_numeric(gp.Size)
print(gp.Size.head())

print(gp.Size.describe().round)

print(gp["Installs"].unique())  #sayıların yanında hem + hemde virgüller var
gp.Installs = gp.Installs.apply(lambda x:x.replace("+",""))
gp.Installs = gp.Installs.apply(lambda x:x.replace(",",""))
gp.Installs = gp.Installs.apply(lambda x:int(x))
print(gp["Installs"].unique())

print(gp["Price"].unique())
gp.Price = gp.Price.apply(lambda x:x.replace("$",""))
gp.Installs = gp.Installs.apply(lambda x:float(x))
print(gp["Price"].unique())


gp["Genres"] = gp["Genres"].str.split(";").str[0]
print(len(gp["Genres"].unique()))
print(gp["Genres"].unique())
print(gp["Genres"].value_counts())  #bir tane olan music audio değerini musice atama

gp["Genres"].replace("Music & Audio","Music",inplace=True)
print(gp["Last_Updated"].head())
gp["Last_Updated"] = pd.to_datetime(gp["Last_Updated"])

print(gp.head())
print(gp.dtypes)

import matplotlib.pyplot as plt
gp["Type"].value_counts().plot(kind="bar",color ="red")  #bar grafik kırmızı
sns.boxplot(x = "Type",y ="Rating", data = gp)
sns.countplot(y= "Content_Rating", data = gp)

plt.title("Content Rating with their counts ")
sns.boxplot(x ="Content_Rating", y = "Rating", data = gp)


































