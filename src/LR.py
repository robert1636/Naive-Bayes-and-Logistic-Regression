# LR solution

# Load libraries
import os
import numpy as np
import math

def sigmoid(scores):
    return 1 / (1 + math.exp(-scores))

# iterate over files in Ham directory in Train folder
# path to training sets
path_train_ham = "../train/ham"
path_train_spam = "../train/spam"
# path for testing files
path_test_ham = "../test/ham"
path_test_spam = "../test/spam"



learning_rate = 0.0001
lamb = 0.001
num_iter = 500
weights = {}

#training
weights["x0"] = float(0)
for step in range(num_iter):
    for filename in os.listdir(path_train_ham):
        d = {}
        f = open(os.path.join("../train/ham" ,filename), "r")
        for line in f:
            for word in line.split():
                if word not in d:
                    d[word] = 1
                else:
                    d[word] += 1
        # make prediction with current weights
        prediction = float(0)
        for word in d:
            if word in weights:
                prediction += weights[word] * d[word]
        prediction_sigmoid = sigmoid(prediction)
        output_error_signal = 1 - prediction_sigmoid
        gradient = 1 * output_error_signal
        weights["x0"] += learning_rate * gradient - learning_rate * lamb * weights["x0"]
        for word in d:
            if word not in weights:
                weights[word] = float(0)
            else:
                gradient = d[word] * output_error_signal
                weights[word] += learning_rate * gradient - learning_rate * lamb * weights[word]
                
    for filename in os.listdir(path_train_spam):
        d = {}
        f = open(os.path.join("../train/spam" ,filename), "r")
        for line in f:
            for word in line.split():
                if word not in d:
                    d[word] = 1
                else:
                    d[word] += 1
        # make prediction with current weights
        prediction = float(0)
        for word in d:
            if word in weights:
                prediction += weights[word] * d[word]
                
        prediction_sigmoid = sigmoid(prediction)
        output_error_signal = 0 - prediction_sigmoid
        gradient = 1 * output_error_signal
        #might be wrong equation
        weights["x0"] += learning_rate * gradient - learning_rate * lamb * weights["x0"]
        for word in d:
            if word not in weights:
                weights[word] = float(0)
            else:
                gradient = d[word] * output_error_signal
                #might be wrong equation
                weights[word] += learning_rate * gradient - learning_rate * lamb * weights[word]

# testing
test_ham = []
for filename in os.listdir(path_test_ham):
    test_d = {}
    f = open(os.path.join("../test/ham" ,filename), "r")
    for line in f:
        for word in line.split():
            if word not in test_d:
                test_d[word] = 1
            else:
                test_d[word] += 1
    test_pred = float(0)
    for word in test_d:
        if word in weights:
            test_pred += weights[word] * test_d[word]
    if test_pred > 0:
        test_ham.append(1)
    else:
        test_ham.append(0)
#print "Ham prediction accuracy with stopwords: ", sum(test_ham) * 1.0 / len(test_ham)
print "Ham prediction accuracy without stopwords: ", "{0:.2%}".format(sum(test_ham) * 1.0 / len(test_ham))

test_spam = []
for filename in os.listdir(path_test_spam):
    test_d = {}
    f = open(os.path.join("../test/spam" ,filename), "r")
    for line in f:
        for word in line.split():
            if word not in test_d:
                test_d[word] = 1
            else:
                test_d[word] += 1
    test_pred = float(0)
    for word in test_d:
        if word in weights:
            test_pred += weights[word] * test_d[word]
    if test_pred < 0:
#        print test_pred
        test_spam.append(1)
    else:
#        print test_pred
        test_spam.append(0)
#print test_spam
#print "Spam prediction accuracy with stopwords: ", sum(test_spam) * 1.0 / len(test_spam)
print "Spam prediction accuracy without stopwords: ", "{0:.2%}".format(sum(test_spam) * 1.0 / len(test_spam))
#print weights
        