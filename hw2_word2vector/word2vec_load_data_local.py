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
# ... load in the data and convert tokens to one-hot indices
# .................................................................................


def loadData(filename):
	global uniqueWords, wordcodes, wordcounts
	override = False
	if override:
		# ... for debugging purposes, reloading input file and tokenizing is quite slow
		# ...  >> simply reload the completed objects. Instantaneous.
		fullrec = pickle.load(open("w2v_fullrec.p", "rb"))
		wordcodes = pickle.load(open("w2v_wordcodes.p", "rb"))
		uniqueWords = pickle.load(open("w2v_uniqueWords.p", "rb"))
		wordcounts = pickle.load(open("w2v_wordcounts.p", "rb"))
		return fullrec

	# ... load in first 15,000 rows of unlabeled data file.  You can load in
	# more if you want later (and should do this for the final homework)
	handle = open(filename, "r", encoding = "utf8")
	fullconts = handle.read().split("\n")
	fullconts = fullconts  # (TASK) Use all the data for the final submission
	# ... apply simple tokenization (whitespace and lowercase)
	# lower cases
	fullconts = [" ".join(fullconts).lower()]

	print("Generating token stream...")
	# ... (TASK) populate fullrec as one-dimension array of all tokens in the order they appear.
	# ... ignore stopwords in this process
	# ... for simplicity, you may use nltk.word_tokenize() to split fullconts.
	# ... keep track of the frequency counts of tokens in origcounts.
	# split into words
	tokens = word_tokenize(fullconts[0])
	stop_words = set(stopwords.words('english'))
	# remove punctuation from each word
	table = str.maketrans('', '', string.punctuation)
	fullrec = []
	fullrec_stop = []
	for word in tokens:
		stripped_word = word.translate(table)
		if stripped_word.isalpha():
			if stripped_word not in stop_words:
				fullrec.append(stripped_word)
				fullrec_stop.append(stripped_word)
			else:
				fullrec_stop.append('<STOP>')
		else:
			pass
	print(len(fullrec))
	print(len(fullrec_stop))

	print("Performing original counting..")
	# ... (TASK) populate array fullrec_filtered to include terms as-is that appeared at least min_count times
	# ... replace other terms with <UNK> token.
	# ... update frequency count of each token in dict wordcounts where: wordcounts[token] = freq(token)

	origcounts = Counter(fullrec)
	print("Performing minimum thresholding..")
	min_count = 50
	wordcounts = Counter()
	for i in range(len(fullrec_stop)):
		if origcounts[fullrec_stop[i]] >= min_count:
			wordcounts[fullrec_stop[i]] += 1
		else:
			fullrec_stop[i] = "<UNK>"
			wordcounts["<UNK>"] += 1

	print("Producing one-hot indicies")
	# ... (TASK) sort the unique tokens into array uniqueWords
	# ... produce their one-hot indices in dict wordcodes where wordcodes[token] = onehot_index(token)
	# ... replace all word tokens in fullrec with their corresponding one-hot indices.
	uniqueWords = list(wordcounts.keys())
	wordcodes = {k: v for v, k in enumerate(uniqueWords)}
	fullrec_filtered = [wordcodes.get(w, len(uniqueWords)) for w in fullrec_stop]
	# print(fullrec_filtered)

	# ... close input file handle
	handle.close()
	# ... store these objects for later.
	# ... for debugging, don't keep re-tokenizing same data in same way.
	# ... just reload the already-processed input data with pickles.
	# ... NOTE: you have to reload data from scratch if you change the min_count, tokenization or number of input rows

	pickle.dump(fullrec_filtered, open("w2v_fullrec.p", "wb+"))
	pickle.dump(wordcodes, open("w2v_wordcodes.p", "wb+"))
	pickle.dump(uniqueWords, open("w2v_uniqueWords.p", "wb+"))
	pickle.dump(dict(wordcounts), open("w2v_wordcounts.p", "wb+"))

	# ... output fullrec should be sequence of tokens, each represented as their one-hot index from wordcodes.
	return fullrec_filtered


# .................................................................................
# ... compute sigmoid value
# .................................................................................
@jit
def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))


# .................................................................................
# ... generate a table of cumulative distribution of words
# .................................................................................


def negativeSampleTable(train_data, uniqueWords, wordcounts, exp_power = 0.75):
	# global wordcodes
	# ... stores the normalizing denominator (count of all tokens, each count raised to exp_power)

	print("Generating exponentiated count vectors")
	# ... (TASK) for each uniqueWord, compute the frequency of that word to the power of exp_power
	# ... store results in exp_count_array.
	exp_count_array = [wordcounts[w] ** exp_power for w in uniqueWords]
	max_exp_count = sum(exp_count_array)

	print("Generating distribution")

	# ... (TASK) compute the normalized probabilities of each term.
	# ... using exp_count_array, normalize each value by the total value max_exp_count so that
	# ... they all add up to 1. Store this corresponding array in prob_dist
	#

	print("Filling up sampling table")
	# ... (TASK) create a dict of size table_size where each key is a sequential number and its value is a one-hot index
	# ... the number of sequential keys containing the same one-hot index should be proportional to its prob_dist value
	# ... multiplied by table_size. This table should be stored in cumulative_dict.
	# ... we do this for much faster lookup later on when sampling from this table.

	cumulative_dict = dict()
	table_size = 1e7
	prob_idx = [round(float(cnt) / float(max_exp_count) * table_size) for cnt in exp_count_array]
	ini = 0
	for i in range(len(prob_idx)):
		for j in range(ini, ini + prob_idx[i]):
			cumulative_dict[j] = i
		ini += prob_idx[i]

	pickle.dump(cumulative_dict, open("w2v_cumulative_dict.p", "wb+"))
	return cumulative_dict


# .................................................................................
# ... generate a specific number of negative samples
# .................................................................................


def generateSamples(context_idx, num_samples):
	global samplingTable, uniqueWords, randcounter
	results = []
	# ... (TASK) randomly sample num_samples token indices from samplingTable.
	# ... don't allow the chosen token to be context_idx.
	# ... append the chosen indices to results
	while len(results) < num_samples:
		sample_idx = random.sample(list(samplingTable.values()), 1)[0]
		if sample_idx not in context_idx:
			results.append(sample_idx)
	return results





# .................................................................................
# ... learn the weights for the input-hidden and hidden-output matrices
# .................................................................................




# .................................................................................
# ... Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
# .................................................................................

def load_model():
	handle = open("saved_W1.data", "rb")
	W1 = np.load(handle)
	handle.close()
	handle = open("saved_W2.data", "rb")
	W2 = np.load(handle)
	handle.close()
	return [W1, W2]


# .................................................................................
# ... Save the current results to an output file. Useful when computation is taking a long time.
# .................................................................................

def save_model(W1, W2):
	handle = open("saved_W1.data", "wb+")
	np.save(handle, W1, allow_pickle = False)
	handle.close()

	handle = open("saved_W2.data", "wb+")
	np.save(handle, W2, allow_pickle = False)
	handle.close()


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


if __name__ == '__main__':
	if len(sys.argv) == 1:
		filename = '/Users/Loielaine/Not Sync/unlabeled-data.txt'
		# filename = sys.argv[1]  # feel free to read the file in a different way

		# ... load in the file, tokenize it and assign each token an index.
		# ... the full sequence of characters is encoded in terms of their one-hot positions

		fullsequence = loadData(filename)
		print("Full sequence loaded...")
		# print(uniqueWords)
		# print (len(uniqueWords))

		# ... now generate the negative sampling table
		print("Total unique words: ", len(uniqueWords))
		print("Preparing negative sampling table")
		samplingTable = negativeSampleTable(fullsequence, uniqueWords, wordcounts)

	# ... we've got the word indices and the sampling table. Begin the training.
	# ... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
	# ... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
	# ... ... and uncomment the load_model() line

	# train_vectors(preload = False)
	# print(np.asarray(word_embeddings).shape)
	# print(np.asarray(proj_embeddings).shape)

	else:
		print("Please provide a valid input filename")
		sys.exit()
