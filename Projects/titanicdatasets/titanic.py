# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 19:58:42 2023

@author: volka
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

tt = pd.read_csv("train.csv")
print(tt.describe())
print(tt.isnull().sum())

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
tt["Age"] = imputer.fit_transform(tt["Age"].values.reshape(-1,1))
tt.drop(["Cabin","Name","Ticket"],axis=1, inplace=True)
tt.dropna(inplace=True)
print(tt.isnull().sum())

for age,Id in tt[["Age","PassengerId"]].values:
        if 0 <= age <= 10.0 :
            tt.at[Id-1,"Age"] = "Child"
        elif age <= 18.0:
            tt.at[Id-1,"Age"] = "Young"
        elif age <= 40.0:
            tt.at[Id-1,"Age"] = "MiddleAge"
        elif age > 40.0:
            tt.at[Id-1,"Age"] = "Old"
            
print(tt["Age"].value_counts())

from sklearn.preprocessing import LabelEncoder
la = LabelEncoder()
tt["Sex"] = la.fit_transform(tt["Sex"].values.reshape(-1,1))

X = tt.iloc[:,2:].values
y = tt["Survived"].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers = [("encoder",OneHotEncoder(),[2,6])],remainder = "passthrough")
X = ct.fit_transform(X)
                                 
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y = train_test_split(X,y, test_size = 0.3) 


from sklearn.metrics import accuracy_score,confusion_matrix
def score(pred,real):
    cm = confusion_matrix(real,pred)
    sc_a = round(accuracy_score(real,pred),3)
    print(f"{cm},\naccuracy score : {sc_a}")
    
    
def test_models():
    classifier1 = DecisionTreeClassifier(criterion = 'entropy')
    classifier1.fit(train_X,train_y)
    DeTree_pred = classifier1.predict(test_X)

    classifier2 = KNeighborsClassifier(n_neighbors=8)
    classifier2.fit(train_X,train_y)
    KNN_pred = classifier2.predict(test_X)

    classifier3 = SVC(kernel='rbf')
    classifier3.fit(train_X,train_y)
    svm_pred = classifier3.predict(test_X)

    classifier4 = RandomForestClassifier(n_estimators=100)
    classifier4.fit(train_X,train_y)
    pred_y = classifier4.predict(test_X)
    
    classifier5 = GaussianNB()
    classifier5.fit(train_X,train_y)
    gnb_pred = classifier5.predict(test_X)
    
    print("==============================================")
    print("Decision Tree test result : ")
    score(DeTree_pred,test_y)
    print("==============================================")
    print("KNN test result : ")
    score(KNN_pred,test_y)
    print("==============================================")
    print("SVC test result : ")
    score(svm_pred,test_y)
    print("==============================================")
    print("Random forest test result : ")
    score(pred_y,test_y)
    print("==============================================")
    print("Gaussian Naive Bayes test result : ")
    score(pred_y,test_y)
    
test_models()
from sklearn.model_selection import cross_val_score

k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
   

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')

#Visualition
import seaborn as sns
plt.figure(figsize=(15,8))
sns.displot(data=tt, x="Age",kde= True)
sns.countplot(x=tt["Survived"])
sns.countplot(data=tt, x="Pclass", hue="Survived")
plt.title('Embarked vs Age', fontsize = 15)
sns.boxplot(x=tt["Pclass"])
sns.barplot(data=tt, x="Age", y="Pclass")

fig = plt.figure(figsize =(10, 7))
plt.pie(tt, labels = "Sex")
    
plt.show()
    
    
    