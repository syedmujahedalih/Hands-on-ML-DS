# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:43:08 2018

@author: admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("mushrooms.csv")
features = data.iloc[:,[5,-2,-1]].values
labels = data.iloc[:,0].values.reshape(-1,1)
feat = pd.DataFrame(features)
label = pd.DataFrame(labels)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le_f = LabelEncoder()
features[:,0] = le_f.fit_transform(features[:,0])
features[:,1] = le_f.fit_transform(features[:,1])
features[:,2] = le_f.fit_transform(features[:,2])

oh_f = OneHotEncoder(categorical_features=[0,1,2])
features = oh_f.fit_transform(features).toarray()

features = features[:,1:]

le_l = LabelEncoder()
labels = le_l.fit_transform(labels).reshape(-1,1)

from sklearn.model_selection import train_test_split
f_train,f_test,l_train,l_test = train_test_split(features,labels,test_size=0.25,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,p=2)
knn.fit(f_train,l_train)
pred1=knn.predict(f_test)
score1 = knn.score(f_test,l_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(l_test,pred1)