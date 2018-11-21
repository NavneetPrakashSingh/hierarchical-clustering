#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 00:14:31 2018
understanding hierarchial clustering
@author: hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data set
dataSet = pd.read_csv("/home/hp/Desktop/HC/Filtered_Smoking_Dataset.csv")
X = dataSet.iloc[:,[0,1]].values

#using dendogram to find the optimim number of cluster
print("plotting first graph")
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title("dendogram")
plt.xlabel("Clusters")
plt.ylabel("Eucledian distance")
plt.show()

#fitting hierarichal clustering to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 3, affinity="euclidean", linkage="ward")
y_hc = hc.fit_predict(X)

#vizualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')

plt.title('Clusters of customers')
plt.xlabel('Age')
plt.ylabel('Health Index (1-10)')
plt.legend()
plt.show()