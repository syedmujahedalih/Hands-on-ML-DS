# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 18:08:36 2018

@author: admin
"""

import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("Foodtruck.csv")
features = data.iloc[:,:-1].values
labels = data.iloc[:,-1].values
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(features_train,labels_train)
labels_pred = regressor.predict(features_test)
score = regressor.score(features_test,labels_test)

plt.scatter(features_train,labels_train,color='red')
plt.plot(features_train,regressor.predict(features_train),color='blue')
plt.title('Profit Estimation')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()

plt.scatter(features_test,labels_test,color='red')
plt.plot(features_train,regressor.predict(features_train),color='blue')
plt.title('Profit Estimation')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()
jaipur = regressor.predict(3.073)
print(jaipur)