#Ryan Gosiaco
import cPickle
import multiprocessing as mp
import numpy as np
import time

fin = open("data_files.pl", "rb");
data = cPickle.load(fin);
Xtrain = data[0]
ytrain = data[1]
Xtest = data[2]
ytest = data[3]

def go_nn(Xtrain, ytrain, Xtest, ytest):
    correct =0
    for i in range(Xtest.shape[0]): ## For all testing instances
        nowXtest = Xtest[i,:]
        ### Find the index of nearest neighbor in training data
        dis_smallest = np.linalg.norm(Xtrain[0,:]-nowXtest) 
        idx = 0
        for j in range(1, Xtrain.shape[0]):
            dis = np.linalg.norm(nowXtest-Xtrain[j,:])
            if dis < dis_smallest:
                dis_smallest = dis
                idx = j
        ### Now idx is the index for the nearest neighbor
        
        ## check whether the predicted label matches the true label
        if ytest[i] == ytrain[idx]:  
            correct += 1
    acc = correct/float(Xtest.shape[0])
    return acc

#Accuracy 0.794000 Time 1037.521343 secs.
#Accuracy 0.794000 Time 907.036880 secs.

#Server 
#Accuracy 0.794000 Time 167.881621 secs.

output = mp.Queue()

def go_nn_multi(Xtrain, ytrain, Xtest, ytest, cores, chunk, output):
    correct = 0
    parts = Xtest.shape[0]/cores

    for i in range(chunk*parts,((chunk+1)*parts)): ## For all testing instances
        nowXtest = Xtest[i,:]
        ### Find the index of nearest neighbor in training data
        dis_smallest = np.linalg.norm(Xtrain[0,:]-nowXtest) 
        idx = 0
        for j in range(1, Xtrain.shape[0]):
            dis = np.linalg.norm(nowXtest-Xtrain[j,:])
            if dis < dis_smallest:
                dis_smallest = dis
                idx = j
        ### Now idx is the index for the nearest neighbor
        
        ## check whether the predicted label matches the true label
        if ytest[i] == ytrain[idx]:  
            correct += 1
    acc = correct/float(parts)
    output.put(acc)

start_time = time.time()

cpus = 4
procs = [mp.Process(target = go_nn_multi, args = (Xtrain, ytrain, Xtest, ytest, cpus, x, output)) for x in range(cpus)]

for p in procs:
    p.start()

for p in procs:
    p.join()

results = [output.get() for p in procs]

#print(results)

#start_time = time.time()
#acc = go_nn_multi(Xtrain, ytrain, Xtest, ytest)
acc = np.average(results)

print("Accuracy %lf Time %lf secs.\n" % (acc, time.time()-start_time))

