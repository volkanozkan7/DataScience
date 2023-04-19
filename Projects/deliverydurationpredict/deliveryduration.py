# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:15:42 2023

@author: volka
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("historical_data.csv")

df.head()
df.info()
df["created_at"] = pd.to_datetime(df["created_at"])
df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"])
df["created_at"].info()

#feature creation: Target variable = Delivery time - order creation time

from datetime import datetime
df["total_delivery_time"] = (df["actual_delivery_time"] - df["created_at"])
df["total_delivery_time"].info()

#create a new feature. Busy deliverer ratio % = Total Busy deliverer / total onshift deliverer
df["busy_deliverer_ratio"] = (df["total_busy_dashers"] / df["total_onshift_dashers"])
#If busy deliverer ratio is higher and lesser the deliverer capacity, delivery duration will be longer.
df["estimated_non_prep_duration"] = (df["estimated_store_to_consumer_driving_duration"] + df["estimated_order_place_duration"])

#Modelling
df["market_id"].nunique()
df["store_id"].nunique()
df["order_protocol"].nunique()

order_protocol_dummies = pd.get_dummies(df.order_protocol)
order_protocol_dummies = order_protocol_dummies.add_prefix("order_protocol_")
order_protocol_dummies.head()

market_id_dummies = pd.get_dummies(df.market_id)
market_id_dummies = market_id_dummies.add_prefix("market_id_")
market_id_dummies.head()

#write the most repeated stores on null rows

store_id_unique = df["store_id"].unique().tolist()
store_id_and_category = {store_id: df[df.store_id == store_id].store_primary_category.mode() for store_id in store_id_unique}

def fill(store_id):
    try:
        return store_id_and_category[store_id].values[0]
    except:
        return np.nan
    
df["nan_free_store_primary_category"] = df.store_id.apply(fill)

store_primary_category_dummies = pd.get_dummies(df.nan_free_store_primary_category)
store_primary_category_dummies = store_primary_category_dummies.add_prefix("category_")
store_primary_category_dummies.head()
    
#drop created_at , market_id , store_id, store_primary_category , actual_delivery_time , actual_total_delivery_duration
df_train = df.drop(columns = ["created_at" , "market_id" , "store_id", "store_primary_category",
                              "actual_delivery_time" ,"nan_free_store_primary_category","order_protocol"])
df_train.head()

df_train = pd.concat([df_train , order_protocol_dummies,market_id_dummies,store_primary_category_dummies],axis = 1)
df_train = df_train.astype("float32")
df_train.head()
df["busy_deliverer_ratio"].describe()
#we have infinite numbers.
np.where(np.any(~np.isfinite(df_train),axis=0) ==True)
df_train.replace([np.inf, -np.inf],np.nan,inplace = True)
df_train.dropna(inplace =True)
df_train.shape
df["busy_deliverer_ratio"].describe()

corr  = df_train.corr()
mask = np.triu(np.ones_like(corr,dtype = bool))
f, ax = plt.subplots(figsize= (11,9))

cmap = sns.diverging_palette(230,20,as_cmap  = True)
sns.heatmap(corr,mask=mask,cmap=cmap, vmax =.3 , center = 0,square = True,linewidth= .5,cbar_kws = {"shrink": .5})
df_train["category_indonesian"].describe()

#drop redundant values
def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0,df.shape[1]):
        for j in range(0, i +1):
            
           pairs_to_drop.add((cols[i],cols[j]))
    return pairs_to_drop

def top_correlations(df,n= 5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending = False)
    return au_corr[0:n]
print("Top Absolute correlations")
print(top_correlations(df_train,20))

# total_onshift_dashers,total_busy_dashers high correlation
df_train = df.drop(columns = ["created_at" , "market_id" , "store_id", "store_primary_category",
                              "total_delivery_time" ,"nan_free_store_primary_category","order_protocol"])
df_train = pd.concat([df_train,order_protocol_dummies,store_primary_category_dummies],axis = 1)
df_train = df.drop(columns = ["total_onshift_dashers","total_busy_dashers","category_indonesian","estimated_non_prep_duration"])
df_train = df_train.astype("float32")
df_train.replace([np.inf , -np.inf],np.nan,inplace= True)
df_train.dropna(inplace = True)
df_train.head()































