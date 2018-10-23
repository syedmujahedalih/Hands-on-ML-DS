# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:16:45 2018

@author: admin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("deliveryfleet.csv")
features = data.iloc[:,1:].values

from sklearn.cluster import KMeans 
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
    
kmeans = KMeans(n_clusters=2,init='k-means++',random_state=0)
y_means=kmeans.fit_predict(features)

plt.scatter(features[y_means==0,0],features[y_means==0,1],s=100,c='red',label='rural drivers')
plt.scatter(features[y_means==1,0],features[y_means==1,1],s=100,c='blue',label='urban drivers')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of Urban and Rural drivers')
plt.xlabel('Distance')
plt.ylabel('Speeding')
plt.legend()
plt.show()

kmeans = KMeans(n_clusters=4,init='k-means++',random_state=0)
y_means=kmeans.fit_predict(features)

plt.scatter(features[y_means==0,0],features[y_means==0,1],s=100,c='red',label='rural drivers')
plt.scatter(features[y_means==1,0],features[y_means==1,1],s=100,c='blue',label='urban drivers')
plt.scatter(features[y_means==2,0],features[y_means==2,1],s=100,c='cyan',label='speeding urban')
plt.scatter(features[y_means==3,0],features[y_means==3,1],s=100,c='magenta',label='speeding rural')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of Urban and Rural drivers')
plt.xlabel('Distance')
plt.ylabel('Speeding')
plt.legend()
plt.show()

