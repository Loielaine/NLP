import os, sys, re, csv
import pickle
from collections import Counter, defaultdict
import numpy as np
import scipy
import math
import random
import nltk
from scipy.spatial.distance import cosine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from numba import jit
import string

# ... (1) First load in the data source and tokenize into one-hot vectors.
# ... Since one-hot vectors are 0 everywhere except for one index, we only need to know that index.


# ... (2) Prepare a negative sampling distribution table to draw negative samples from.
# ... Consistent with the original word2vec paper, this distribution should be exponentiated.


# ... (3) Run a training function for a number of epochs to learn the weights of the hidden layer.
# ... This training will occur through backpropagation from the context words down to the source word.

# ... (4) Test your model. Compare cosine similarities between learned word vectors.

# ... (5) Feel free make minor change to the structure or input of the skeleton code (P.S you still need to finish all the task and implemenent word2vec correctly)


# .................................................................................
# ... global variables
# .................................................................................


random.seed(10)
np.random.seed(10)
randcounter = 10
np_randcounter = 10

vocab_size = 0
hidden_size = 100
uniqueWords = [""]  # ... list of all unique tokens
wordcodes = {}  # ... dictionary mapping of words to indices in uniqueWords
wordcounts = Counter()  # ... how many times each token occurs
samplingTable = []  # ... table to draw negative samples from


# .................................................................................
# ... Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
# .................................................................................

def load_model():
	handle = open("saved_W1_target.data", "rb")
	W1 = np.load(handle)
	handle.close()
	handle = open("saved_W2_target.data", "rb")
	W2 = np.load(handle)
	handle.close()
	return [W1, W2]


# ... so in the word2vec network, there are actually TWO weight matrices that we are keeping track of. One of them represents the embedding
# ... of a one-hot vector to a hidden layer lower-dimensional embedding. The second represents the reversal: the weights that help an embedded
# ... vector predict similarity to a context word.


# .................................................................................
# ... code to start up the training function.
# .................................................................................
word_embeddings = []
proj_embeddings = []


def train_vectors(preload = False):
	global word_embeddings, proj_embeddings
	if preload:
		[word_embeddings, proj_embeddings] = load_model()
	else:
		curW1 = None
		curW2 = None
		[word_embeddings, proj_embeddings] = trainer(curW1, curW2)
		save_model(word_embeddings, proj_embeddings)


# .................................................................................
# ... find top 10 most similar words to a target word
# .................................................................................
def get_wordscodes(target_word):
	global uniqueWords, wordcodes
	if target_word in uniqueWords:
		target_index = wordcodes[target_word]
	else:
		target_index = wordcodes["<UNK>"]
	return target_index


def normalize(word_vec):
	norm = np.linalg.norm(np.asarray(word_vec))
	if norm == 0:
		return word_vec
	return word_vec / norm


def get_neighbors(target_word):
	global word_embeddings, uniqueWords, wordcodes
	# ... (TASK) search through all uniqueWords and for each token, compute its similarity to target_word.
	# ... you will compute this using the absolute cosine similarity of the word_embeddings for the word pairs.
	# ... Note that the cosine() function from scipy.spatial.distance computes a DISTANCE so you need to convert that to a similarity.
	# ... return a list of top 10 most similar words in the form of dicts,
	# ... each dict having format: {"word":<token_name>, "score":<cosine_similarity>}
	pred_num = 10
	target_index = get_wordscodes(target_word)
	# print(target_index)
	target_vector = word_embeddings[target_index]
	similarity = dict()
	for i in range(len(uniqueWords)):
		# print(uniqueWords[i])
		idx = wordcodes[uniqueWords[i]]
		if idx == target_index:
			continue
		word_vector = word_embeddings[idx]
		word_similarity = 1 - cosine(target_vector, word_vector)
		similarity[uniqueWords[i]] = word_similarity
	sorted_similarity = sorted(similarity.items(), key = lambda x: x[1], reverse = True)
	pred_len = min(pred_num, len(sorted_similarity))
	return dict(sorted_similarity[0:pred_len])


def get_similairy(word1, word2):
	word1_index = get_wordscodes(word1)
	word2_index = get_wordscodes(word2)
	word1_vector = word_embeddings[word1_index]
	word2_vector = word_embeddings[word2_index]
	# word1_vector = proj_embeddings[word1_index]
	# word2_vector = proj_embeddings[word2_index]
	# print(cosine(word1_vector, word2_vector))
	similarity = 1 - cosine(word1_vector, word2_vector)
	# dot = np.dot(word1_vector, word2_vector)
	# norma = np.linalg.norm(word1_vector)
	# normb = np.linalg.norm(word2_vector)
	# similarity = dot / (norma * normb)
	return similarity


# .................................................................................
# ... for the averaged morphological vector combo, estimate the new form of the target word
# .................................................................................


def morphology(word_seq):
	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
	embeddings = word_embeddings
	vectors = [word_seq[0],  # suffix averaged
	           embeddings[wordcodes[word_seq[1]]]]
	vector_math = vectors[0] + vectors[1]


# ... find whichever vector is closest to vector_math
# ... (TASK) Use the same approach you used in function prediction() to construct a list
# ... of top 10 most similar words to vector_math. Return this list.


# .................................................................................
# ... for the triplet (A,B,C) find D such that the analogy A is to B as C is to D is most likely
# .................................................................................

def analogy_word(word_seq):
	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
	embeddings = word_embeddings
	if word_seq[0] in uniqueWords and word_seq[1] in uniqueWords and word_seq[2] in uniqueWords:
		vectors = [embeddings[wordcodes[word_seq[0]]],
		           embeddings[wordcodes[word_seq[1]]],
		           embeddings[wordcodes[word_seq[2]]]]
		vector_math = -vectors[0] + vectors[1] - vectors[2]  # + vectors[3] = 0
		# ... find whichever vector is closest to vector_math
		# ... (TASK) Use the same approach you used in function prediction() to construct a list
		# ... of top 10 most similar words to vector_math. Return this list.
		similarity = dict()
		for i in range(len(uniqueWords)):
			# print(uniqueWords[i])
			idx = wordcodes[uniqueWords[i]]
			word_vector = normalize(word_embeddings[idx, :])
			word_similarity = 1 - cosine(vector_math, word_vector)
			similarity[uniqueWords[i]] = word_similarity
		analogy = sorted(similarity.items(), key = lambda x: x[1], reverse = True)
	else:
		print("Invalid input for analogy.")
		analogy = []
	pred_len = min(10, len(analogy))
	return dict(analogy[0:pred_len])


def analogy_proj(word_seq):
	global word_embeddings, proj_embeddings, uniqueWords, wordcodes
	embeddings = proj_embeddings
	if word_seq[0] in uniqueWords and word_seq[1] in uniqueWords and word_seq[2] in uniqueWords:
		vectors = [embeddings[wordcodes[word_seq[0]]],
		           embeddings[wordcodes[word_seq[1]]],
		           embeddings[wordcodes[word_seq[2]]]]
		vector_math = -vectors[0] + vectors[1] - vectors[2]  # + vectors[3] = 0
		# ... find whichever vector is closest to vector_math
		# ... (TASK) Use the same approach you used in function prediction() to construct a list
		# ... of top 10 most similar words to vector_math. Return this list.
		similarity = dict()
		for i in range(len(uniqueWords)):
			# print(uniqueWords[i])
			idx = wordcodes[uniqueWords[i]]
			word_vector = normalize(word_embeddings[idx, :])
			word_similarity = 1 - cosine(vector_math, word_vector)
			similarity[uniqueWords[i]] = word_similarity
		analogy = sorted(similarity.items(), key = lambda x: x[1], reverse = True)
	else:
		print("Invalid input for analogy.")
		analogy = []
	pred_len = min(10, len(analogy))
	return dict(analogy[0:pred_len])


if __name__ == '__main__':
	if len(sys.argv) == 1:

		# fullsequence = pickle.load(open("./load_data/w2v_fullrec.p", "rb"))
		wordcodes = pickle.load(open("./load_data/w2v_wordcodes.p", "rb"))
		uniqueWords = pickle.load(open("./load_data/w2v_uniqueWords.p", "rb"))
		wordcounts = pickle.load(open("./load_data/w2v_wordcounts.p", "rb"))
		print("Full sequence loaded...")
		print("Total unique words: ", len(uniqueWords))

		train_vectors(preload = True)
		print(np.asarray(word_embeddings).shape)
		print(np.asarray(proj_embeddings).shape)

		# ... we've got the trained weight matrices. Now we can do some predictions
		# ...pick ten words you choose
		targets = ["good", "bad", "food", "apple", 'tasteful', 'unbelievably', 'uncle', 'tool', 'think']
		targ_similarity = []
		for targ in targets:
			print("Target: ", targ)
			bestpreds = get_neighbors(targ)
			targ_similarity.append([targ, bestpreds])
			for k, v in bestpreds.items():
				print(k, ":", v)
			print("\n")
		with open('prob7_output.txt', 'w') as output:
			for i in targ_similarity:
				output.write("%s\n" % i)
		output.close()
		#
		# # ... try an analogy task. The array should have three entries, A,B,C of the format: A is to B as C is to ?
		# print(analogy(["apple", "fruit", "banana"]))
		analogy_inputs = [["apple", "fruit", "banana"],
		                  ['happy', 'smile', "sad"],
		                  ["boy", "girl", 'king'],
		                  ['father', 'dad', 'mother'],
		                  ['father', 'mother', 'dad'],
		                  ['apple', 'apples', 'pear'],
		                  ['good', 'bad', 'great'],
		                  ['ask', 'answer', 'think']
		                  ]
		for analogy_input in analogy_inputs:
			print(analogy_input)
			print('Word embeddings:')
			bestanalogy1 = analogy_word(analogy_input)
			for k, v in bestanalogy1.items():
				print(k, ":", v)
			print('Project embeddings:')
			bestanalogy2 = analogy_proj(analogy_input)
			for k, v in bestanalogy2.items():
				print(k, ":", v)
			print("\n")
		#
		# # ... try morphological task. Input is averages of vector combinations that use some morphological change.
		# # ... see how well it predicts the expected target word when using word_embeddings vs proj_embeddings in
		# # ... the morphology() function.
		# # ... this is the optional task, if you don't want to finish it, common lines from 545 to 556
		#
		# # s_suffix = [word_embeddings[wordcodes["banana"]] - word_embeddings[wordcodes["bananas"]]]
		# # others = [["apples", "apple"], ["values", "value"]]
		# # for rec in others:
		# # 	s_suffix.append(word_embeddings[wordcodes[rec[0]]] - word_embeddings[wordcodes[rec[1]]])
		# # s_suffix = np.mean(s_suffix, axis = 0)
		# # print(morphology([s_suffix, "apples"]))
		# # print(morphology([s_suffix, "pears"]))
		#
		testfilename = 'intrinsic-test.csv'
		testoutput = 'intrisic-output_local_target.csv'
		with open(testfilename) as test, open(testoutput, 'w') as output:
			output.write('id,sim\n')
			line = test.readline()
			while True:
				line = test.readline().lstrip().rstrip()
				if not line:
					break
				else:
					line_id, word1, word2 = line.split(',')
					# print(line_id)
					# print(len(word1))
					# print(len(word2))
					similarity = get_similairy(word1, word2)
					# print(similarity)
					output.write(str(line_id) + ',' + str(similarity) + '\n')
		test.close()
		output.close()

		print(get_similairy('happy', 'sad'))
		print(get_similairy('happy', 'cheerful'))


	else:
		print("Please provide a valid input filename")
		sys.exit()
