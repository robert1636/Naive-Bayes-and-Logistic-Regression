# NB solution

# Load libraries
import os
import math


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
            if word not in new_words:
                new_words.add(word)
            if word not in ham_words:
                ham_words.add(word)
    num_train_ham += 1
for filename in os.listdir(path_train_spam):
    f = open(os.path.join("../train/spam" ,filename), "r")
    for line in f:
        for word in line.split():
            if word not in new_words:
                new_words.add(word)
            if word not in spam_words:
                spam_words.add(word)
    num_train_spam += 1
#print len(new_words)
#print num_train_ham
#print num_train_spam
    
# define two 2D arrays to hold words, with column as word and row as file
#words_ham = [[1] * len(new_words) for _ in range(num_train_ham)]
#words_spam = [[1] * len(new_words) for _ in range(num_train_spam)]

# initialized every word as 1 already, need further thinking here!!!

words_ham = dict((key, 1) for key in ham_words)
words_spam = dict((key, 1) for key in spam_words)
words_whole = dict((key, 1) for key in new_words)
# looping through the files again to count the number of apperance of each words
for filename in os.listdir(path_train_ham):
    f = open(os.path.join("../train/ham" ,filename), "r")
    for line in f:
        for word in line.split():
            words_ham[word] += 1
            words_whole[word] += 1
            
for filename in os.listdir(path_train_spam):
    f = open(os.path.join("../train/spam" ,filename), "r")
    for line in f:
        for word in line.split():
            words_spam[word] += 1
            words_whole[word] += 1

#print words_ham
#print words_spam
#print words_whole
#print len(words_whole)
# testing
# path for testing files
path_test_ham = "../test/ham"
path_test_spam = "../test/spam"

# priors
prob_ham = 1.0 * num_train_ham / (num_train_spam + num_train_ham)
prob_spam = 1.0 * num_train_spam / (num_train_spam + num_train_ham)

#print prob_ham
#print prob_spam

# conditional probabilities
ham_words_totalCount = sum(words_ham.values())
spam_words_totalCount = sum(words_spam.values())
#print ham_words_totalCount 
#print spam_words_totalCount
prob_words_ham = dict()
prob_words_spam = dict()
#print words_ham
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
            # problems could be here
            if word in words_whole:
                if word in words_ham:
                    prob_ham_this += math.log(prob_words_ham[word])
                else:
                    prob_ham_this += math.log(1.0 / ham_words_totalCount)
                if word in words_spam:
                    prob_spam_this += math.log(prob_words_spam[word])
                else:
                    prob_spam_this += math.log(1.0 / spam_words_totalCount)
#    print prob_ham_this, prob_spam_this
    if prob_ham_this >= prob_spam_this:
        prob_each_word_ham.append(1)
    else:
        prob_each_word_ham.append(0)

#print sum(prob_each_word)
#print len(prob_each_word)
#print "Ham prediction accuracy: ", sum(prob_each_word_ham) * 1.0 / len(prob_each_word_ham)
print "Ham prediction accuracy with stopwords: ", "{0:.2%}".format(sum(prob_each_word_ham) * 1.0 / len(prob_each_word_ham))
prob_each_word_spam = []       
for filename in os.listdir(path_test_spam):
    f = open(os.path.join("../test/spam" ,filename), "r")
    # choosing a class
    # for each file, go through the file and count each word's frequency
    prob_ham_this = prob_ham
    prob_spam_this = prob_spam
    for line in f:
        for word in line.split():
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
#print sum(prob_each_word)
#print len(prob_each_word)         
#print "Spam prediction accuracy: ", 1 - sum(prob_each_word_spam) * 1.0 / len(prob_each_word_spam)
print "Spam prediction accuracy with stopwords: ", "{0:.2%}".format(sum(prob_each_word_spam) * 1.0 / len(prob_each_word_spam))





