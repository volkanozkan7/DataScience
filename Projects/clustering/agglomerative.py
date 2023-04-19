# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 19:52:08 2023

@author: volka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("musteriler.csv")

X = veriler.iloc[:,3:].values  #maaş ve hacim


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3 , init = "k-means++")  # 3 merkez noktası vericek
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar = []
for i in range(1,10):
    kmeans = KMeans(n_clusters = i,init = "k-means++",random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    

plt.plot(range(1,10),sonuclar)  #grafikten 2,3,4 gibi k değerleri alabiliceğini gördük
plt.show()

kmeans = KMeans(n_clusters = 4,init = "k-means++",random_state=123)
Y_tahmin = kmeans.fit_predict(X)
print(Y_tahmin)
plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100,c="red")
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100,c="blue")
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100,c="green")
plt.title("kmeans")
plt.show()

#HC

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters = 3,affinity = "euclidean",linkage="ward")  #affinity = manhattan,euclidean, cosine, precomputed
#linkage = ward,complete,avarage
Y_tahmin = ac.fit_predict(X)
print(Y_tahmin)

plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100,c="red")
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100,c="blue")
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100,c="green")
plt.title("agglomerative")
plt.show()