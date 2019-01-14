#Ryan Gosiaco
import cPickle
import numpy as np
import sklearn as sk
import time
import multiprocessing as mp
from sklearn import datasets

fin = open("data_files.pl", "rb");
data = cPickle.load(fin);
X_array = data[0]
y = data[1]

X_ar = sk.preprocessing.normalize(X_array,axis=0)

""" vec = gradientDescent(X_ar, y_norm, 0.001, 200, 0.01)
print(vec)
print("Time %lf secs.\n" % (time.time()-start_time)) """

def gradient_m(X, y, w):
    inner = y * np.dot(np.transpose(w), X)
    bottom = 1 + np.exp(inner)
    first = (-y/bottom) * X
    final = np.add(first/X.shape[0], w[0])
    return final
    #return final[:,1]

def gradientDescent(X, y, epsilon, iterations, ss):
    w_0 = np.zeros((X.shape[1],1))
    r_0 = np.linalg.norm(gradient_m(X[0,:], y[0], w_0))
    #step_size = 0.01
    step_size = ss

    for i in range(iterations):
        print(i)
        for j in range(y.shape[0]):
            grad = gradient_m(X[j,:], y[j], w_0)
            w_0 = w_0 - (step_size * grad)
            if np.linalg.norm(grad) < (epsilon * r_0):
                return w_0[:,1]
    return w_0[:,1]

#start_time = time.time()
#vec = gradientDescent(X_ar, y, 0.001, 200, 0.01)
#print(vec)
#print("Time %lf secs.\n" % (time.time()-start_time))

""" w_0 = np.zeros((X_ar.shape[1],1))
grad = gradient_m(X_ar[0,:], y[0], w_0)
print(grad) """

def gradientUpdate(X, y, w, step_size):
    grad = gradient_m(X, y, w)
    w = w - (step_size * grad)
    return w[:,1]

pool = mp.Pool(processes = 4)

def gradientDescent_m(X, y, epsilon, iterations, ss):
    w_0 = np.zeros((X.shape[1],1))
    r_0 = np.linalg.norm(gradient_m(X[0,:], y[0], w_0))
    #step_size = 0.01
    step_size = ss

    for i in range(iterations):
        print(i)
        results = [pool.apply(gradientUpdate, args=(X[j,:], y[j], w_0, ss)) for j in range(y.shape[0])]

    return results

start_time = time.time()
vec = gradientDescent_m(X_ar, y, 0.001, 200, 0.01)
print(vec)
print("Time %lf secs.\n" % (time.time()-start_time))