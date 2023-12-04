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

def calcSimilarity(unseenAttrib, seenAttrib):
    #calculating the similarity
    similarity = np.dot(unseenAttrib, seenAttrib.T)
    simSum = np.sum(similarity, axis=1)
    #normalising the similarity vector
    similarity = similarity/simSum.reshape(similarity.shape[0],1)
    return similarity

def meanUnseen(similarity, mSeen):
    mUnseen = np.dot(similarity,mSeen)
    return mUnseen

def trainFeatureScale(seenInput, u, featureScale):
    
    diff = u - seenInput
    sq = np.sqrt(np.sum(np.square(diff), axis=0).reshape(diff.shape[1], 1))
    
    featureScale = featureScale - 0.1*featureScale*sq
    featureScale/=np.sum(featureScale)

    return featureScale

def learning_with_prototype():
    mSeen = meanSeen(X_seen)

    featureScale = np.ones((mSeen.shape[1], 1))
    featureScale/=np.sum(mSeen.shape[1])

    # learning featureScale k number of times, change accordingly for different accuracy
    k = 15
    for j in range(k):
        for i in range(X_seen.shape[0]):
            featureScale = trainFeatureScale(X_seen[i], mSeen[i], featureScale)

    # Test Class
    
    s = calcSimilarity(class_attributes_unseen, class_attributes_seen)
    mUnseen = meanUnseen(s, mSeen)
    
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

    print("The accuracy of Method 1 is", 100*accuracy)

# Call the classifier
learning_with_prototype()


