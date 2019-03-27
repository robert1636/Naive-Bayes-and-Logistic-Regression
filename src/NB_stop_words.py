# NB solution

# Load libraries
import os
import math
import re

stop_list = set()
content = open("../stop_words.txt", "r")
for line in content:
    stop_list.add(re.compile("[^a-zA-Z']+").sub(' ', line).strip())

# iterate over files in Ham directory in Train folder
# path to training sets
path_train_ham = "../train/ham"
path_train_spam = "../train/spam"
# number of files in ham and spam folders
num_train_ham = 0
num_train_spam = 0
# number of different words in both files
new_words = set()
ham_words = set()
spam_words = set()
for filename in os.listdir(path_train_ham):
    f = open(os.path.join("../train/ham" ,filename), "r")
    for line in f:
        for word in line.split():
            if word not in stop_list:
                if word not in new_words:
                    new_words.add(word)
                if word not in ham_words:
                    ham_words.add(word)
    num_train_ham += 1
for filename in os.listdir(path_train_spam):
    f = open(os.path.join("../train/spam" ,filename), "r")
    for line in f:
        for word in line.split():
            if word not in stop_list:
                if word not in new_words:
                    new_words.add(word)
                if word not in spam_words:
                    spam_words.add(word)
    num_train_spam += 1

# initialized every word as 1 already, need further thinking here!!!
words_ham = dict((key, 1) for key in ham_words)
words_spam = dict((key, 1) for key in spam_words)
words_whole = dict((key, 1) for key in new_words)
# looping through the files again to count the number of apperance of each words
for filename in os.listdir(path_train_ham):
    f = open(os.path.join("../train/ham" ,filename), "r")
    for line in f:
        for word in line.split():
            if word not in stop_list:
                words_ham[word] += 1
                words_whole[word] += 1
            
for filename in os.listdir(path_train_spam):
    f = open(os.path.join("../train/spam" ,filename), "r")
    for line in f:
        for word in line.split():
            if word not in stop_list:
                words_spam[word] += 1
                words_whole[word] += 1

# testing
# path for testing files
path_test_ham = "../test/ham"
path_test_spam = "../test/spam"

# priors
prob_ham = 1.0 * num_train_ham / (num_train_spam + num_train_ham)
prob_spam = 1.0 * num_train_spam / (num_train_spam + num_train_ham)

# conditional probabilities
ham_words_totalCount = sum(words_ham.values())
spam_words_totalCount = sum(words_spam.values())
prob_words_ham = dict()
prob_words_spam = dict()

for key in words_ham:
    prob_words_ham[key] = words_ham[key] * 1.0 / ham_words_totalCount
    
for key in words_spam:
    prob_words_spam[key] = words_spam[key] * 1.0 / spam_words_totalCount

# loop through each file in ham folder and calculate the probablity using NB
prob_each_word_ham = []
for filename in os.listdir(path_test_ham):
    f = open(os.path.join("../test/ham" ,filename), "r")
    # choosing a class
    # for each file, go through the file and count each word's frequency
    prob_ham_this = prob_ham
    prob_spam_this = prob_spam
    for line in f:
        for word in line.split():
            if word not in stop_list:
                if word in words_whole:
                    if word in words_ham:
                        prob_ham_this += math.log(prob_words_ham[word])
                    else:
                        prob_ham_this += math.log(1.0 / ham_words_totalCount)
                    if word in words_spam:
                        prob_spam_this += math.log(prob_words_spam[word])
                    else:
                        prob_spam_this += math.log(1.0 / spam_words_totalCount)
    if prob_ham_this >= prob_spam_this:
        prob_each_word_ham.append(1)
    else:
        prob_each_word_ham.append(0)
#print "Ham prediction accuracy: ", sum(prob_each_word_ham) * 1.0 / len(prob_each_word_ham)
print "Ham prediction accuracy without stopwords: ", "{0:.2%}".format(sum(prob_each_word_ham) * 1.0 / len(prob_each_word_ham))
prob_each_word_spam = []       
for filename in os.listdir(path_test_spam):
    f = open(os.path.join("../test/spam" ,filename), "r")
    # choosing a class
    # for each file, go through the file and count each word's frequency
    prob_ham_this = prob_ham
    prob_spam_this = prob_spam
    for line in f:
        for word in line.split():
            if word not in stop_list:
                if word in words_whole:
                    if word in words_ham:
                        prob_ham_this += math.log(prob_words_ham[word])
                    else:
                        prob_ham_this += math.log(1.0 / ham_words_totalCount)
                    if word in words_spam:
                        prob_spam_this += math.log(prob_words_spam[word])
                    else:
                        prob_spam_this += math.log(1.0 / spam_words_totalCount)
    if prob_spam_this >= prob_ham_this:
        prob_each_word_spam.append(1)
    else:
        prob_each_word_spam.append(0)       
#print "Spam prediction accuracy: ", sum(prob_each_word_spam) * 1.0 / len(prob_each_word_spam)
print "Spam prediction accuracy without stopwords: ", "{0:.2%}".format(sum(prob_each_word_spam) * 1.0 / len(prob_each_word_spam))





