# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 14:44:05 2023

@author: volka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("fifa21.csv", sep = ";")
data.head()
data.isnull().sum()
data.info()
data.dtypes

data["nationality"].value_counts().nunique()

#Visualition
plt.figure(figsize = (18,7))
data["nationality"].value_counts().head(12).plot.bar(color = "green")
plt.title("Players from Different Countries")
plt.xlabel("Country")
plt.ylabel("Count")
plt.show()

age  = data.age
plt.figure(figsize = (12,8))
ax = sns.distplot(age,bins = 55,kde = False,color ='red')
ax.set_ylabel('Number of Players')
ax.set_xlabel('Age')
ax.set_title('Distribution of Age of Players')
plt.show()

plt.figure(figsize = (20,7))
data['team'].value_counts().head(15).plot.bar(color = 'orangered')
plt.title('Most Popular Clubs in FIFA-2021')
plt.xlabel('Clubs')
plt.ylabel('Count')
plt.show()

data.sort_values("age",ascending = True)[["name","team","nationality","age"]].head(15)
data.sort_values("age",ascending = False)[["name","team","nationality","age"]].head(5)
data.groupby(["team"])["age"].mean().sort_values(ascending= True).head(11)
data.groupby(["team"])["age"].mean().sort_values(ascending= False).head(11)

data.groupby(["name"])["overall"].mean().sort_values(ascending= False).head(11)
data[data['position'] == 'ST'][['name', 'age', 'team', 'nationality']].head(10)

plt.figure(figsize=(12,8))
sns.heatmap(data[['age', 'nationality', 'overall', 'potential', 'team', 'position']].corr(), annot = True)
plt.title('Overall relation between columns of the Dataset', fontsize = 17)
plt.show()

def player(x):
    return data.loc[data["name"]== x]
def club(x):
    return data[data["team"] == x][['name','overall','potential','position','age']]

player("Cristiano Ronaldo")
club('Liverpool ')

plt.figure(figsize=(15,8))
sns.lineplot(data['overall'], data['age'])
plt.title('Overall vs Age', fontsize = 15)
sns.boxplot(x=data["age"])

sns.barplot(data=data, x="age", y="overall")












