# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 23:32:21 2023

@author: volka
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import mode

df = pd.read_csv('test-set.csv', header = 0)
df.head(5) # look for first 5 rows
df.shape # print size of dataframe
is_NaN = df.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = df[row_has_NaN]
rows_with_NaN
df = df.fillna("Unknown") # fill NaNs with "Unknown"
df.head(3) # look for first 3 rows
def plot_stats(fig, subplot_id, column, color_map, rotation):
    fig.add_subplot(subplot_id)
    color_len = len(df[column].unique())
    color = color_map(np.linspace(0, 1, color_len))
    count_classes = df[column].value_counts()
    plt.title(column)
    count_classes.plot(kind='bar', color=color)
    plt.xticks(rotation=rotation)
    # plt.ylabel("Count")
    
fig = plt.figure(figsize=(25, 20))
plot_stats(fig, 331, "Gender", plt.cm.bwr, "horizontal")
plot_stats(fig, 332, "Married", plt.cm.cool, "horizontal")
plot_stats(fig, 333, "Graduated", plt.cm.winter, "horizontal")
plot_stats(fig, 334, "Profession", plt.cm.Wistia, "vertical")
plot_stats(fig, 335, "WorkExperience", plt.cm.copper, "vertical")
plot_stats(fig, 336, "SpendingScore", plt.cm.rainbow, "horizontal")
plot_stats(fig, 337, "FamilySize", plt.cm.jet, "horizontal")
plot_stats(fig, 338, "Category", plt.cm.turbo, "vertical")
plot_stats(fig, 339, "Segmentation", plt.cm.nipy_spectral, "horizontal")
plt.show()


bins = [0, 
        max(df['Age'])/7, 
        2*max(df['Age'])/7, 
        3*max(df['Age'])/7, 
        4*max(df['Age'])/7, 
        5*max(df['Age'])/7, 
        6*max(df['Age'])/7, 
        max(df['Age'])] 

time_intervals = pd.cut(df['Age'], bins=bins)


# and then we group data by class
df_grouped2 = df.groupby(['Gender', time_intervals]).size().reset_index(name='Count')

# creating masks
mask1 = df_grouped2['Gender']=="Female"
mask2 = df_grouped2['Gender']=="Male"

# applying masks
df_sliced1 = df_grouped2.loc[mask1]
df_sliced2 = df_grouped2.loc[mask2]

# creating plot with 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,3))

# setting colormaps for each subplot
color1 = plt.cm.summer(np.linspace(0, 1, len(df_sliced1['Age'].unique())))
color2 = plt.cm.Wistia(np.linspace(0, 1, len(df_sliced1['Age'].unique())))

# drawing plots
df_sliced1.plot(x='Age', y='Count', kind = 'bar', color=color1, title='Female', ax=axes[0])
df_sliced2.plot(x='Age', y='Count', kind = 'bar', color=color2, title='Male', ax=axes[1])

plt.show()

df["Gender"] = pd.factorize(df["Gender"])[0]
df["Married"] = pd.factorize(df["Married"])[0]
df["Graduated"] = pd.factorize(df["Graduated"])[0]
df["Profession"] = pd.factorize(df["Profession"])[0]
df["SpendingScore"] = pd.factorize(df["SpendingScore"])[0]
df["Category"] = pd.factorize(df["Category"])[0]


df["WorkExperience"] = df["WorkExperience"].replace("Unknown",df["WorkExperience"].mode()[0])
df["FamilySize"] = df["FamilySize"].replace("Unknown",df["FamilySize"].mode()[0])
df.head(5) # look for first 5 rows

scaler = MinMaxScaler(feature_range=(0, 1)) # range is [0, 1]
normed = scaler.fit_transform(df.copy().drop(columns=["CustomerID", "Segmentation"]))
df_normed = pd.DataFrame(data=normed, columns=df.columns[1:-1])
df_normed.head() # check the result

# take x-values from df
x_columns = df_normed.columns

# create 'x' data for train
X_train = df_normed[x_columns]

# convert to numpy
X_train = X_train.to_numpy()

K = len(df['Segmentation'].unique()) # number of clusters
ITERS = 1000 # maximum iterations
RUNS = 10 # total runs

# init k-means method
kmeans = KMeans(n_clusters=K, random_state=0, max_iter=ITERS, n_init=RUNS, verbose=False)

# fit and predict labels
X_train1 = X_train.copy()
labels = kmeans.fit_predict(X_train1)
print("- labels = ", labels)

# count labels
count_labels = np.bincount(labels)
print("- count_labels = ", count_labels)

# find centroids
centroids = kmeans.cluster_centers_
print("- centroids.shape = ", centroids.shape)

# name cols as in initial dataset (without last "Segmentation")
cols = df_normed.columns[:-1]
df_res = df_normed[cols]
# add result labels
df_res["Cluster"] = labels
# insert "CustomerID" for easier understanding
df_res.insert(0, "CustomerID", df["CustomerID"])
# check the results
df_res.head()

# get unique labels (found clusters id's)
u_labels = np.unique(labels)
plt.figure(figsize=(15,10))
for i in u_labels:
    mask = (df_res["Cluster"] == i)
    plt.plot(df_res[mask]["CustomerID"], ".", label = i)
plt.title('CustomerID and corresponding clusters')
plt.legend()
plt.show()

# reduce data only to 2 features
pca = PCA(n_components=2).fit_transform(X_train.copy())
# chech result shape
pca.shape

print(pca[:3])

# initialize k-means
kmeans = KMeans(n_clusters=K, random_state=0, max_iter=ITERS, n_init=RUNS, verbose=False)

# predict the labels of clusters
labels = kmeans.fit_predict(pca)
print("- labels = ", labels)

pca_res = pd.DataFrame(pca.copy())
pca_res["Cluster"] = labels
pca_res.insert(0, "CustomerID", df["CustomerID"])
pca_res.head()

# get the centroids
centroids = kmeans.cluster_centers_
u_labels = np.unique(labels)

# plot the results:
plt.figure(figsize=(15,10))
for i in u_labels:
    mask = (pca_res["Cluster"] == i)
    plt.scatter(pca_res.loc[mask][0], pca_res.loc[mask][1], label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 100, color = 'k', marker="^")
plt.title('Clusters of Automobile Customers')
plt.legend()
plt.show()

