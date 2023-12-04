import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

X_seen=np.load('data/X_seen.npy',encoding='bytes',allow_pickle=True) # (40 x N_i x D): 40 feature matrices. X_seen[i] is the N_i x D feature matrix of seen class i
Xtest=np.load('data/Xtest.npy',encoding='bytes',allow_pickle=True)	# (6180, 4096): feature matrix of the test data.
Ytest=np.load('data/Ytest.npy',encoding='bytes',allow_pickle=True)	# (6180, 1): ground truth labels of the test data
class_attributes_seen=np.load('data/class_attributes_seen.npy')	# (40, 85): 40x85 matrix with each row being the 85-dimensional class attribute vector of a seen class.
class_attributes_unseen=np.load('data/class_attributes_unseen.npy')	# (10, 85): 10x85 matrix with each row being the 85-dimensional class attribute vector of an  unseen class.

def meanSeen(seenClass):
    mSeen = np.zeros((seenClass.shape[0], seenClass[0].shape[1]))
    for i in range(0, seenClass.shape[0]):
        mSeen[i] = (np.mean(seenClass[i], axis=0)).reshape(1, seenClass[0].shape[1])
    return mSeen

def meanUnseen(mSeen, As, Aus, lmbda):
    W = np.dot(np.linalg.inv(np.dot(As.T, As) + lmbda*(np.identity(As.shape[1]))), np.dot(As.T, mSeen))
    mUnseen = np.dot(Aus, W)
    return mUnseen

def classify(mUnseen, Xtest, Ytest, featureScale):
    accuracy = 0
    dist = np.zeros((Ytest.shape[0], mUnseen.shape[0]))
    for i in range(mUnseen.shape[0]):
        diff = mUnseen[i] - Xtest
        sq = np.square(diff)
        d = np.dot(sq, featureScale)
        dist[:, i] = d.reshape(d.shape[0],)

    y_pred = np.argmin(dist, axis=1)
    y_pred = y_pred.reshape(y_pred.shape[0],1)
    y_pred+=1 #to match one based labelling for Ytest
    accuracy = 1 - np.count_nonzero(y_pred-Ytest)/float(Ytest.shape[0])

    return accuracy

def linear_classifier():
    mSeen = meanSeen(X_seen)

    featureScale = np.ones((mSeen.shape[1], 1))
    featureScale/=np.sum(mSeen.shape[1])

    lmbdaArr = [0.01, 0.1, 1, 10, 20, 50, 100]

    for i in lmbdaArr:
        mUnseen = meanUnseen(mSeen, class_attributes_seen, class_attributes_unseen, i)
        accuracy = classify(mUnseen, Xtest, Ytest, featureScale)
        print("The accuracy for lambda = ", i,"is", 100*accuracy)
    

linear_classifier()


