# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 18:10:35 2023

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
    kmeans = KMeans(n_clusters = i,init = "k-means++",random_state=1)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    

plt.plot(range(1,10),sonuclar)  #grafikten 2,3,4 gibi k değerleri alabiliceğini gördük

from sklearn.metrics import silhouette_score
silhouettes = []
ks  = list(range(2,12))

for n_cluster in ks:
    kmeans = KMeans(n_clusters = n_cluster).fit(X)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X, label,metric="euclidean")
    print("For n_clusters={}, The Silhouette Coefficient is {}" . format(n_cluster,sil_coeff))
    silhouettes.append(sil_coeff)    
    
plt.figure(figsize = (12,8))
plt.subplot(211)
plt.scatter(ks,silhouettes,marker = "x", c="r")
plt.plot(ks,silhouettes)