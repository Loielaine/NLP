#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10:50 AM 1/19/20

@author: Yixi Li
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score
from collections import Counter
import matplotlib.pyplot as plt
import re


# def tokenize(line):
# 	# line = [re.split(r'[.:/\\;\-_<>\[\]=+{}()\"?*&^%$#@!~]+', words) for words in line]
# 	# line = line.split()
# 	words = [w for words in line for w in words]
# 	words = [w.rstrip('\'').lstrip('\'').lower() for w in words if w]
# 	return words

def tokenize(line):
	words = line.split()
	# words = re.split(r'[.:/\\;\-_<>\[\]=+{}()\"?*&^%$#@!~]+', line)
	# word_pattern = re.compile("[a-zA-Z]+")
	# words = list(filter(word_pattern.match, words))
	words = list(set(words))
	return words


def count_word_toal(wordlist):
	wordlist_flat = [w for words in wordlist for w in words]
	cnt = Counter()
	for word in wordlist_flat:
		cnt[word] += 1
	filtered_cnt = {key: value for (key, value) in cnt.items() if value > 2}
	return filtered_cnt


def count_word(features, wordlist, n, p):
	x = np.zeros((n, p))
	for i in range(n):
		# print('counting words for %d' % i)
		cnti = Counter(wordlist[i])
		value = [cnti.get(w, 0) for w in features]
		value = np.asarray(value)
		x[i, :] = value
	return x


# def sphere(x):
# 	n, p = x.shape
# 	x_sd = np.std(x, axis = 0)
# 	# print(x_sd.shape)
# 	# identity = np.identity(p)
# 	column = np.ones(n).reshape((n, 1))
# 	x_bar = np.mean(x, axis = 0).reshape((1, p))
# 	x_sphere = (x - column.dot(x_bar)).dot(np.diag(1 / x_sd))
# 	# print(x_sphere.shape)
# 	return x_sphere
#
#
# def spheretest(x):
# 	n, p = x.shape
# 	column = np.ones(n).reshape((n, 1))
# 	return (x - column.dot(x_bar)).dot(np.diag(1 / x_sd))


def read_data(X_train, y_train):
	wordlist = []
	n = 0
	with open(X_train) as text:
		while True:
			line = text.readline()
			if not line:
				break
			else:
				words = tokenize(line)
				wordlist.append(words)
				n += 1
	text.close()

	with open('output/wordlist_lr_v0.txt', 'w') as output:
		for w in wordlist:
			output.write('%s\n' % w)

	filtered_cnt = count_word_toal(wordlist)
	features = filtered_cnt.keys()
	# features = list(set([w for words in wordlist for w in words]))
	p = len(features)
	print('N: %d' % n)
	print('d: %d' % p)
	x = count_word(features, wordlist, n, p)
	# x_sphere = sphere(x)
	with open(y_train) as label:
		y_train = label.readlines()
	label.close()
	y = list(map(int, y_train))
	y = np.asarray(y).reshape((n, 1))
	# return x, y, features
	return x, y, features


def sigmoid(x):
	return 1. / (1. + np.exp(-x))


def log_likelihood(x, y, w):
	ll = np.sum(y * (x.dot(w)) - np.log(1. + np.exp(x.dot(w))))
	return ll


# def log_likelihood(xrand, yrand, w):
# 	ll = yrand * (xrand.dot(w)) - np.log(1. + np.exp(xrand.dot(w)))
# 	return ll


def compute_gradient(xrand, yrand, w):
	g = (yrand - sigmoid(xrand.dot(w))) * xrand.T
	# print(g.shape)
	return g.reshape((-1, 1))


def logistic_regression(x, y, learning_rate, num_step):
	n, p = x.shape
	x0 = np.ones((n, 1))
	xnew = np.hstack((x0, x))
	w = np.zeros((p + 1, 1))
	# w_step = np.zeros(shape = (p + 1, num_step))
	ll_step = []
	for j in range(num_step):
		print('number of step: %d' % j)
		i = np.random.randint(0, n)
		xrand = xnew[i, :]  # 1*(p+1)
		yrand = y[i]
		w_update = w + learning_rate * compute_gradient(xrand, yrand, w)
		w = np.copy(w_update)
		ll = log_likelihood(xnew, y, w)
		ll_step.append(ll)
	# print('log likelihood for step %d is %f' % (j, ll))
	# if len(ll_step) >= 2 and ll - ll_step[-2] < 10 ** (-5):
	# 	break
	# print(w.shape)
	# w_step[:, j] = w.reshape(-1)
	return w.reshape((p + 1, 1)), ll_step


def predict(X_test, features, w):
	wordlist = []
	n = 0
	with open(X_test) as text:
		while True:
			line = text.readline()
			if not line:
				break
			else:
				words = tokenize(line)
				wordlist.append(words)
				n += 1
	text.close()
	p = len(features)
	x_test = count_word(features, wordlist, n, p)
	# x_test = spheretest(x_test)
	x0 = np.ones((n, 1))
	x_test_new = np.hstack((x0, x_test))
	y = sigmoid(x_test_new.dot(w))
	y_predict = []
	for predicted in y:
		if predicted > 0.5:
			y_predict.append(1)
		else:
			y_predict.append(0)
	# print(y_predict)
	return y_predict


# def getfonescore(y_predict, y_test):
# 	with open(y_test, 'r') as test:
# 		y_test = test.readlines()
# 	test.close()
# 	y_test = list(map(int, y_test))
# 	return f1_score(y_test, y_predict, average = 'binary')

def getscore(y_predict, y_test):
	with open(y_test, 'r') as test:
		y_test = test.readlines()
	test.close()
	y_test = list(map(int, y_test))
	false_index = []
	for i in range(len(y_test)):
		if y_test[i] != y_predict[i]:
			false_index.append(i)
		else:
			continue
	return f1_score(y_test, y_predict), precision_score(y_test, y_predict)


X_train = '../data/X_train.txt'
y_train = '../data/y_train.txt'
X_test = '../data/X_test.txt'
X_dev = '../data/X_dev.txt'
y_dev = '../data/y_dev.txt'
method_name = 'SGD_LR_v6'

x, y, features = read_data(X_train, y_train)

w_list = []
ll_list = []
fscore_list = []
learning_rate_list = [5 * 10 ** (-3)]
num_step = 10**7
for learning_rate in learning_rate_list:
	w_step, ll_step = logistic_regression(x, y, learning_rate, num_step)
	w_list.append(w_step)
	ll_list.append(ll_step)
	y_predict = predict(X_dev, features, w_step)
	# fonescore = getfonescore(y_predict, y_dev)
	fonescore, precision = getscore(y_predict, y_dev)
	print(fonescore)
	print(precision)
	fscore_list.append(fonescore)

for i in range(len(learning_rate_list)):
	plt.plot(range(len(ll_list[i])), ll_list[i], label = str(learning_rate_list[i]))
plt.xlabel('steps')
plt.ylabel('log likelihood for each step')
plt.legend()
plt.savefig('output/' + method_name + '.png')

id_best = fscore_list.index(max(fscore_list))
ll_best = learning_rate_list[id_best]
fscore_best = fscore_list[id_best]
print('The best learning rate is %f' % ll_best)
print('The best f1 score is %f' % fscore_best)
w_best = w_list[id_best]
y_test_predict = predict(X_test, features, w_best)
with open('output/y_test_predict_' + method_name + str(id_best) + '.csv', 'w') as output:
	output.write('Id,Category\n')
	for i in range(len(y_test_predict)):
		output.write(str(i) + ',' + str(y_test_predict[i]) + '\n')
