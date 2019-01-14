#Ryan Gosiaco
#Problem 1

import pickle
import numpy as np
import time

X = pickle.load(open('data_dense.pl','rb'))
n,p = X.shape
init_centers = np.zeros((10,p))
init_centers[0:10,:] = X[0:10,:]
centers = init_centers 

#Helper Functions
def euclidean_distance(a,b):
    return np.linalg.norm(a-b)

def assign_cluster(a,centers):
    dists = np.array([euclidean_distance(a,x) for x in centers])
    return [np.argmin(dists), np.min(dists)]

def cluster_mean(cluster):
    return np.mean(cluster,axis=0)

start_time = time.time()

counter = 0

for m in range(40):
    objective = 0
    clusters = [[] for i in range(10)]
    for i in range(n):
        a = X[i,:]
        c, min_dist = assign_cluster(a, centers)
        clusters[c].append(a)
        objective = objective + min_dist**2
        
    centers = [] 
    for i in range(10):
        centers.append(cluster_mean(clusters[i]))
    centers = np.array(centers)
    #x_center = centers[:,0]; y_center = centers[:,1]
    if (counter == 10):
        print "At iteration", m, "the objective is", objective
        counter = 0
    counter += 1
print "At iteration", 40, "the objective is", objective
print "Time:",time.time() - start_time


