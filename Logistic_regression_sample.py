# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 18:04:03 2018

@author: admin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("affairs.csv")
features = data.iloc[:,:-1].values
labels = data.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
oh = OneHotEncoder(categorical_features=[6,7])
le = LabelEncoder()

features = oh.fit_transform(features).toarray()
features = features[:,1:]

from sklearn.model_selection import train_test_split
f_train,f_test,l_train,l_test = train_test_split(features,labels,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(f_train,l_train)
l_pred = classifier.predict(f_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(l_test,l_pred)
X = [0,1,0,0,0,0,0,0,0,1,0,0,3,25,3,1,4,16]
Y = np.array(X).reshape(1,-1)
Y = Y[:,1:]
new = classifier.predict_proba(Y)



