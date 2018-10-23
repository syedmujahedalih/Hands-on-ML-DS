# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 18:05:22 2018

@author: admin
"""

import pandas as pd
import numpy as np

data = pd.read_csv("PastHires.csv")

features = data.iloc[:,:-1].values
feat = pd.DataFrame(features)

labels = data.iloc[:,-1].values.reshape(-1,1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

features[:,1] = le.fit_transform(features[:,1])
features[:,3] = le.fit_transform(features[:,3])
features[:,4] = le.fit_transform(features[:,4])
features[:,5] = le.fit_transform(features[:,5])
labels = le.fit_transform(labels).reshape(-1,1)

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(features,labels)

x = np.array([5,1,2,1,1,0]).reshape(1,6)
y = dtr.predict(x)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)

regressor.fit(features,labels)

x1 = np.array([10,1,4,0,1,0]).reshape(1,-1)
x2 = np.array([10,0,4,1,0,0]).reshape(1,-1)
y1 = regressor.predict(x1)
y2 = regressor.predict(x2)