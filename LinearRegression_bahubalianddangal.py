# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 18:35:06 2018

@author: admin
"""
import matplotlib.pyplot as plt
import pandas as pd
data = pd.read_csv("Bahubali2_vs_Dangal.csv")
features = data.iloc[:,0].values.reshape(-1,1)
labels_b = data.iloc[:,1].values
labels_d = data.iloc[:,2].values
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(features,labels_b)
pred_b = regressor1.predict(features)
day10_b = regressor1.predict(10)
print(day10_b)
regressor2 = LinearRegression()
regressor2.fit(features,labels_d)
pred_d = regressor2.predict(features)
day10_d = regressor2.predict(10)
print(day10_d)

plt.scatter(features,labels_b,color='red')
plt.plot(features,pred_b,color='blue')
plt.title('Baahubali Stats')
plt.xlabel('Days')
plt.ylabel('Earning')
plt.show()

plt.scatter(features,labels_d,color = 'red')
plt.plot(features,pred_d,color='blue')
plt.title('Dangal Stats')
plt.xlabel('Days')
plt.ylabel('Earning')
plt.show()
