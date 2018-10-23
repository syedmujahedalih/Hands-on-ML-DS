# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 17:50:45 2018

@author: admin
"""

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("bluegills.csv")
features = data.iloc[:,0:1].values
labels = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
f_train,f_test,l_train,l_test = train_test_split(features,labels,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg_1 = LinearRegression()
reg_1.fit(f_train,l_train)
pred1 = reg_1.predict(f_test)
score1 = reg_1.score(f_train,l_train)

plt.scatter(f_train,l_train,color='red')
plt.plot(f_test,pred1,color='blue')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
f_train_poly = poly_reg.fit_transform(f_train)
poly_reg.fit(f_train,l_train)
reg_2 = LinearRegression()
reg_2.fit(f_train_poly,l_train)
reg_2.predict(poly_reg.fit_transform(f_test))
score2 = reg_2.score(f_train_poly,l_train)

reg_2.predict(poly_reg.fit_transform(5))