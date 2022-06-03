import os
import pickle
import random
import operator

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
from collections import defaultdict


# function to get the distance between feature vectors and find neighbors
def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + \
            distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])

    return neighbors


# identify the class of the instance
def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1

    sorter = sorted(classVote.items(),
                    key=operator.itemgetter(1), reverse=True)

    return sorter[0][0]


# function to evaluate the model
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1

    return (1.0 * correct) / len(testSet)


# directory that holds the wav files
directory = "E:\\owner-project\\music-genre-classification\\data\\genres_original\\"

# binary file where we will collect all the features
# extracted using mfcc (Mel Frequency Cepstral Coefficients)
f = open("my.dat", 'wb')

i = 0

for folder in os.listdir(directory):
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory+folder):
        try:
            (rate, sig) = wav.read(directory+folder+"/"+file)
            mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, i)
            pickle.dump(feature, f)
        except Exception as e:
            print('Got an exception: ', e, ' in folder: ',
                  folder, ' filename: ', file)

f.close()


# Split the dataset into training and testing sets respectively
dataset = []


def loadDataset(filename, split, trSet, teSet):
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])


trainingSet = []
testSet = []
loadDataset('my.dat', 0.66, trainingSet, testSet)


def distance(instance1, instance2, k):
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2-mm1).transpose(),
                 np.linalg.inv(cm2)), mm2-mm1))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance


# making predictions using KNN
leng = len(testSet)
predictions = []
for x in range(leng):
    predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))

accuracy1 = getAccuracy(testSet, predictions)
print(accuracy1)


# testing the code with external samples

test_dir = "E:\\owner-project\\music-genre-classification\\test\\"
# test_file = test_dir + "test1.wav"
# test_file = test_dir + "test2.wav"
# test_file = test_dir + "test3.wav"
# test_file = test_dir + "test4.wav"
# test_file = test_dir + "test5.wav"
test_file = test_dir + "test6.wav"
# test_file = test_dir + "test7.wav"

(rate, sig) = wav.read(test_file)
mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
feature = (mean_matrix, covariance, i)


results = defaultdict(int)

i = 1
for folder in os.listdir(directory):
    results[i] = folder
    i += 1

pred = nearestClass(getNeighbors(dataset, feature, 5))
print(results[pred])
