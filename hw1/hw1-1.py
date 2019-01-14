#Ryan Gosiaco
import numpy as np
import sklearn as sk
from sklearn import datasets
filename = "E:\STA141C\HW1\cpusmall.txt"
X,y = datasets.load_svmlight_file(filename)

from scipy import sparse
X_array = sparse.csr_matrix.todense(X)

def gradient(X, y, w):
    first = np.transpose(np.dot(np.transpose(X), X))
    second = np.transpose(np.dot(np.transpose(X), y))
    third = np.transpose(np.dot(first, w))
    final = np.add(np.subtract(third, second), w)
    return final[:,1]

def gradientDescent(X, y, epsilon, iterations, ss):
    w_0 = np.zeros((X.shape[1],1))
    r_0 = np.linalg.norm(gradient(X, y, w_0))
    #step_size = 0.01
    step_size = ss
    for i in range(iterations):
        grad = gradient(X, y, w_0)
        w_0 = w_0 - (step_size * grad)
        #w_0 = np.subtract(w_0, np.multiply(step_size, grad))
        if np.linalg.norm(grad) < (epsilon * r_0):
            #print(np.linalg.norm(grad))
            #print(epsilon * r_0)
            #print(i)
            return w_0[:,1]
    return w_0[:,1]

X_ar = sk.preprocessing.normalize(X_array,axis=0)
y_norm = sk.preprocessing.normalize(np.reshape(y, (-1,1)), axis=0)

#print(gradientDescent(X_ar, y_norm, 0.001, 200, 0.01)) #0.06769835
#print(gradientDescent(X_ar, y_norm, 0.001, 200, 0.001)) #0.04512519
#print(gradientDescent(X_ar, y_norm, 0.001, 200, 0.0001)) #0.00702038
#print(gradientDescent(X_ar, y_norm, 0.001, 200, 0.00001)) #0.00073691
#print(gradientDescent(X_ar, y_norm, 0.001, 200, 0.000001)) #7.40524083e-05
#print(gradientDescent(X_ar, y_norm, 0.001, 200, 0.0000001)) #7.40886683e-06

def mse(X, y, w):
    total = 0
    for i in range(len(X)):
        x_i = X[i,:]
        y_i = y[i,:]
        square = np.square(np.dot(np.transpose(x_i), w) - y_i)
        total = total + square
    return np.divide(total, len(X))

from sklearn.model_selection import KFold
def cross_validation(X, y):
    total = 0
    kf = KFold(n_splits=5)
    
    for train_ind, test_ind in kf.split(X):
        #grad = gradientDescent(X, y, 0.001, 200, 0.01)
        weights = gradientDescent(X[train_ind], y[train_ind], 0.001, 200, 0.01)
        mse_i = mse(X[test_ind], y[test_ind], weights)
        total = total + mse_i
    print(total/5)

#cross_validation(X_ar, y_norm)#5.53147692e-05

from scipy.sparse import csr_matrix

def gradient_csr(X, y, w):
    X_trans = csr_matrix.transpose(X)
    first = csr_matrix.transpose(np.dot(X_trans, X))
    second = csr_matrix.transpose(np.dot(X_trans, y))
    third = csr_matrix.transpose(np.dot(first, w))
    final = np.add(np.subtract(third, second), w)
    return final[:,1]

def gradientDescent_csr(X, y, epsilon, iterations, ss):
    w_0 = np.zeros((X.shape[1],1))
    r_0 = np.linalg.norm(gradient_csr(X, y, w_0))
    #step_size = 0.01
    step_size = ss
    for i in range(iterations):
        grad = gradient_csr(X, y, w_0)
        w_0 = w_0 - (step_size * grad)
        #w_0 = np.subtract(w_0, np.multiply(step_size, grad))
        if np.linalg.norm(grad) < (epsilon * r_0):
            #print(np.linalg.norm(grad))
            #print(epsilon * r_0)
            #print(i)
            return w_0[:,1]
    return w_0[:,1]

X_ez_test,y_ez_test = datasets.load_svmlight_file("E:\STA141C\HW1\E2006.test")
X_ez_te_ar = sparse.csr_matrix.todense(X_ez_test)

X_ez_train,y_ez_train = datasets.load_svmlight_file("E:\STA141C\HW1\E2006.train")
X_ez_tr_ar = sparse.csr_matrix.todense(X_ez_train)

def E2006(X_train, y_train, X_test, y_test):
    weights = gradientDescent_csr(X_train, y_train, 0.001, 200, 0.01)
    mse_i = mse(X_test, y_test, weights)
    print(mse_i)

E2006(X_ez_tr_ar, y_ez_train, X_ez_te_ar, y_ez_test)