#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 3:35 PM 1/18/20

@author: Yixi Li
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score
import matplotlib.pyplot as plt
import re


def tokenize(line):
	words = line.split()
	# words = re.split(r'[.:/\\;\-_<>\[\]=+{}()\"?*&^%$#@!~]+', line)
	# word_pattern = re.compile("[a-zA-Z]+")
	# words = list(filter(word_pattern.match, words))
	return words


def countword(words, wordlist):
	for w in set(words):
		if w in wordlist.keys():
			wordlist[w] += 1
		else:
			wordlist[w] = 1
	# print(max(cnt.values()))
	return wordlist


def probword(wordlist, length, smoothing_alpha):
	prob = dict()
	for key, value in wordlist.items():
		prob[key] = (float(value) + smoothing_alpha) / (float(len(wordlist.keys())) + smoothing_alpha * length)
	return prob


def train(X_train, y_train, smoothing_alpha):
	wordlist = dict()
	wordlist_0 = dict()
	wordlist_1 = dict()
	linecnt = 0
	cnt_y0 = 0
	cnt_y1 = 1
	with open(X_train) as text, open(y_train) as label:
		while (linecnt < 10000):
			line = text.readline()
			words = tokenize(line)
			wordlist = countword(words, wordlist)

			linelabel = int(label.readline())
			if linelabel == 0:
				cnt_y0 += 1
				wordlist_0 = countword(words, wordlist_0)
			else:
				cnt_y1 += 1
				wordlist_1 = countword(words, wordlist_1)
			linecnt += 1
	text.close()
	label.close()
	N = len(wordlist.keys())
	prob_y0 = float(cnt_y0) / float(cnt_y0 + cnt_y1)
	prob_y1 = float(cnt_y1) / float(cnt_y0 + cnt_y1)
	# prob_y0 = (float(cnt_y0) + smoothing_alpha) / (float(cnt_y0 + cnt_y1) + smoothing_alpha)
	# prob_y1 = 1 - prob_y0
	prob_x = probword(wordlist, N, smoothing_alpha)
	prob_x0 = probword(wordlist_0, N, smoothing_alpha)
	prob_x1 = probword(wordlist_1, N, smoothing_alpha)
	# print(sum(prob_x0.values()))
	# print(sum(prob_x1.values()))
	smooth_0 = smoothing_alpha / (float(len(wordlist_0.keys())) + smoothing_alpha * N)
	smooth_1 = smoothing_alpha / (float(len(wordlist_1.keys())) + smoothing_alpha * N)
	return prob_y0, prob_y1, prob_x, prob_x0, prob_x1, smooth_0, smooth_1


def loglikelihood(obs, prob_x0, prob_x1, smooth_0, smooth_1, smoothing_alpha):
	ll_y0 = ll_y1 = 1.0
	for i in range(len(obs)):
		ll_y0 *= prob_x0.get(obs[i], smooth_0)
		ll_y1 *= prob_x1.get(obs[i], smooth_1)
	return ll_y0, ll_y1


def classify(prob_y0, prob_y1, prob_x0, prob_x1, smooth_0, smooth_1, smoothing_alpha, X_test):
	prior_y0 = prob_y0
	prior_y1 = prob_y1
	y_predict = []
	with open(X_test) as text:
		while True:
			line = text.readline()
			if not line:
				break
			else:
				words = tokenize(line)
				posterior_y0 = loglikelihood(words, prob_x0, prob_x1, smooth_0, smooth_1, smoothing_alpha)[0] * prior_y0
				posterior_y1 = loglikelihood(words, prob_x0, prob_x1, smooth_0, smooth_1, smoothing_alpha)[1] * prior_y1
				if posterior_y0 >= posterior_y1:
					y_predict.append(0)
				else:
					y_predict.append(1)
	text.close()
	return y_predict


# def getscore(y_predict, y_test):
# 	with open(y_test, 'r') as test:
# 		y_test = test.readlines()
# 	test.close()
# 	y_test = list(map(int, y_test))
# 	false_index = []
# 	for i in range(len(y_test)):
# 		if y_test[i] != y_predict[i]:
# 			false_index.append(i)
# 		else:
# 			continue
# 	return f1_score(y_test, y_predict), precision_score(y_test, y_predict), false_index

def getfonescore(y_predict, y_test):
	with open(y_test, 'r') as test:
		y_test = test.readlines()
	test.close()
	y_test = list(map(int, y_test))
	return f1_score(y_test, y_predict, average = 'binary')


"""
with open('output/wordlist.txt', 'w') as output:
	for w in wordlist:
		output.write('%s\n' % wordlist)
"""

smoothing_alpha = 0.0
smoothing_alpha_list = np.arange(0.0, 1.0, 0.05)
X_train = '../data/X_train.txt'
y_train = '../data/y_train.txt'
X_test = '../data/X_test.txt'
X_dev = '../data/X_dev.txt'
y_dev = '../data/y_dev.txt'
method_name = 'tokenizer_v3'

fonescore_list = []
for smoothing_alpha in smoothing_alpha_list:
	prob_y0, prob_y1, prob_x, prob_x0, prob_x1, smooth_0, smooth_1 = train(X_train, y_train, smoothing_alpha)
	y_predict = classify(prob_y0, prob_y1, prob_x0, prob_x1, smooth_0, smooth_1, smoothing_alpha, X_dev)
	fonescore = getfonescore(y_predict, y_dev)
	# print(fonescore)
	fonescore_list.append(fonescore)

# prob_y0, prob_y1, prob_x, prob_x0, prob_x1, smooth_0, smooth_1 = train(X_train, y_train, smoothing_alpha)
# y_predict = classify(prob_y0, prob_y1, prob_x0, prob_x1, smooth_0, smooth_1, smoothing_alpha, X_dev)
# fonescore, precision, false_index = getscore(y_predict, y_dev)
# print(fonescore)
# print(precision)
# with open('output/y_dev_false_index' + method_name + '.csv', 'w') as output:
# 	for i in range(len(false_index)):
# 		output.write(str(false_index[i]) + '\n')

print(smoothing_alpha_list)
print(fonescore_list)
plt.plot(smoothing_alpha_list, fonescore_list)
plt.xlabel('smoothing alphas')
plt.ylabel('f1 scores')
plt.savefig('output/' + method_name + '.png')

prob_y0, prob_y1, prob_x, prob_x0, prob_x1, smooth_0, smooth_1 = train(X_train, y_train, smoothing_alpha)
y_test_predict = classify(prob_y0, prob_y1, prob_x0, prob_x1, smooth_0, smooth_1, smoothing_alpha, X_test)
with open('output/y_test_predict_' + method_name + '.csv', 'w') as output:
	output.write('Id,Category\n')
	for i in range(len(y_test_predict)):
		output.write(str(i) + ',' + str(y_test_predict[i]) + '\n')
