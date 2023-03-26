# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 20:37:25 2023

@author: volka
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

nt = pd.read_csv("netflix_titles.csv")
print(nt.head())
print(nt.shape)
print(nt.isnull().sum())  #boş değerler

print(nt.nunique())

data = nt.copy()
print(data.shape)

data = data.dropna() #eksik verileri attık
print(data)

'''sns.countplot(nt["type"])
sns.countplot(nt["rating"]).set_xticklabels(sns.countplot(nt["rating"]).get_xticklabels(),rotation=90,ha="right")
fig = plt.gcf()
fig.set_size_inches(20,20)
plt.title("rating")
'''
plt.figure(figsize=(10,8))
sns.countplot(x="rating",hue="type",data=nt)
plt.title("tür ve reyting oranı") 
plt.show()

labels=["Movie","TV Show"]
size=nt["type"].value_counts()
colors = plt.cm.Wistia(np.linspace(0,1,2))
explode=[0,0.1]
plt.rcParams["figure.figsize"] = (9,9)
plt.pie(size,labels=labels,colors=colors,explode=explode,shadow=True,startangle=90)
plt.title("Tür Görünümü",fontsize=25)
plt.legend()
plt.show()

plt.subplots(figsize=(25,15))
wordcloud = WordCloud(background_color="white",width=1920,height=1080).generate(" ".join(data.cast))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("cast.png")
plt.show()