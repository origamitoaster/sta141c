#Ryan Gosiaco
#Problem 2

import pickle
import numpy as np
import time
from scipy.sparse.linalg import norm as nm

X = pickle.load(open('data_sparse_E2006.pl','rb'))
n,p = X.shape
init_centers = np.zeros((10,p))
init_centers[0:10,:] = X[0:10,:].todense()
centers = init_centers

def cluster_mean(cluster):
    return np.mean(cluster,axis=0)

start_time = time.time()

counter = 0

for m in range(40):
    objective = 0
    cluster = [[] for i in range(10)]

    B_squareds = np.zeros(10)
    for k in range(10):
        b = centers[k]
        B_squareds[k] = np.linalg.norm(b)**2

    for i in range(n):
        distances = np.zeros(10)
        a = X[i,:] #row
        A_squared = nm(a)**2 #scalar

        for k in range(10):
            b = centers[k] #scalar
            TwoAB = 2 * a.dot(b) #array
            B_squared = B_squareds[k] # scalar
            euclidean_distance = A_squared - TwoAB + B_squared
            distances[k] = euclidean_distance
        c = np.argmin(distances)
        cluster[c].append(i)
        objective = objective + np.min(distances)
    
    for k in range(10):
        indices = cluster[k]
        subset = X[indices, :]
        centers[k,:] = subset.mean(0)
    if (counter == 10):
        print "At iteration", m, "the objective is", objective
        counter = 0
    counter += 1
print "At iteration", 40, "the objective is", objective
print "Time:", time.time()-start_time
