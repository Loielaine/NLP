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
# ... compute sigmoid value
# .................................................................................
@jit
def sigmoid(x):
	return 1.0 / (1 + np.exp(-x))


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
		sample = random.randint(0, len(samplingTable) - 1)
		sample_idx = samplingTable[sample]
		if sample_idx not in context_idx:
			results.append(sample_idx)
	return results


@jit(nopython = True)
def performDescent(num_samples, learning_rate, center_token, context_words, W1, W2, negative_indices):
	# sequence chars was generated from the mapped sequence in the core code

	# ... (TASK) implement gradient descent. Find the current context token from context_words
	# ... and the associated negative samples from negative_indices. Run gradient descent on both
	# ... weight matrices W1 and W2.
	# ... compute the total negative log-likelihood and store this in nll_new.
	# ... You don't have to use all the input list above, feel free to change them
	nll_new = 0

	w_j = np.empty(num_samples + 1, dtype = np.int64)
	t_j = np.zeros(num_samples + 1, dtype = np.int64)
	t_j[0] = 1

	for k in range(0, len(context_words)):
		w_j[0] = context_words[k]
		w_j[1:] = negative_indices[k]

		h = W1[center_token]

		update_i = np.zeros((hidden_size, len(w_j)))
		for i in range(0, len(w_j)):
			v_j = W2[w_j[i]]
			update_i[:, i] = (sigmoid(np.dot(v_j.T, h)) - t_j[i]) * v_j
			W2[w_j[i]] = v_j - learning_rate * (sigmoid(np.dot(v_j.T, h)) - t_j[i]) * h  # creates v_j_new
		W1[center_token] = h - learning_rate * np.sum(update_i, axis = 1)

		update_nll = np.zeros(len(w_j))
		for i in range(1, len(w_j)):
			update_nll[i - 1] = np.log(sigmoid(-np.dot(W2[w_j[i]].T, h)))  # h is updated in memory
		nll_new = -np.log(sigmoid(np.dot(W2[w_j[0]].T, h))) - update_nll.sum()
	return nll_new, W1, W2


# .................................................................................
# ... learn the weights for the input-hidden and hidden-output matrices
# .................................................................................


def trainer(curW1 = None, curW2 = None):
	global uniqueWords, wordcodes, fullsequence, vocab_size, hidden_size, np_randcounter, randcounter
	vocab_size = len(uniqueWords)  # ... unique characters
	hidden_size = 100  # ... number of hidden neurons
	context_window = [-2, -1, 1,
	                  2]  # ... specifies which context indices are output. Indices relative to target word. Don't include index 0 itself.
	nll_results = []  # ... keep array of negative log-likelihood after every 1000 iterations

	# ... determine how much of the full sequence we can use while still accommodating the context window
	start_point = int(math.fabs(min(context_window)))
	end_point = len(fullsequence) - (max(max(context_window), 0))
	mapped_sequence = fullsequence

	# ... initialize the weight matrices. W1 is from input->hidden and W2 is from hidden->output.
	if curW1 == None:
		np_randcounter += 1
		W1 = np.random.uniform(-.5, .5, size = (vocab_size, hidden_size))
		W2 = np.random.uniform(-.5, .5, size = (vocab_size, hidden_size))
	else:
		# ... initialized from pre-loaded file
		W1 = curW1
		W2 = curW2

	# ... set the training parameters
	epochs = 5
	num_samples = 2
	learning_rate = 0.05
	nll = 0

	# ... Begin actual training
	for j in range(0, epochs):
		print("Epoch: ", j)
		prevmark = 0
		iternum = 0
		# ... For each epoch, redo the whole sequence...
		for i in range(start_point, end_point):
			# ... (TASK) determine which token is our current input. Remember that we're looping through mapped_sequence
			center_token = fullsequence[i]
			iternum += 1
			# print(iternum)
			# ... (TASK) don't allow the center_token to be <UNK>. move to next iteration if you found <UNK>.
			if center_token == wordcodes['<UNK>'] or center_token == len(uniqueWords):
				continue

			whole_context_index = []
			# ... now propagate to each of the context outputs
			for k in range(0, len(context_window)):
				# ... (TASK) Use context_window to find one-hot index of the current context token.
				context_index = fullsequence[i + context_window[k]]
				if context_index < len(uniqueWords):
					whole_context_index.append(context_index)

			if len(whole_context_index) == 0:
				continue
			# ... construct some negative samples
			negative_indices = generateSamples(whole_context_index, len(whole_context_index) * num_samples)
			# print(negative_indices)

			# ... (TASK) You have your context token and your negative samples.
			# ... Perform gradient descent on both weight matrices.
			# ... Also keep track of the negative log-likelihood in variable nll.
			nll, W1, W2 = performDescent(len(whole_context_index) * num_samples, learning_rate, center_token,
			                             whole_context_index, W1, W2, negative_indices)

			# print(W1[center_token])
			# print(W2[center_token])
			if (float(i) / len(mapped_sequence)) >= (prevmark + 0.1):
				print("Progress: ", round(prevmark + 0.1, 1))
				prevmark += 0.1
			if iternum % 10000 == 0:
				print("Negative likelihood: ", nll)
				nll_results.append(nll)

	for nll_res in nll_results:
		print(nll_res)
	return [W1, W2]


# .................................................................................
# ... Load in a previously-saved model. Loaded model's hidden and vocab size must match current model.
# .................................................................................

def load_model():
	handle = open("/home/liyixi/hw2/saved_W1_cluster.data", "rb")
	W1 = np.load(handle)
	handle.close()
	handle = open("/home/liyixi/hw2/saved_W2_cluster.data", "rb")
	W2 = np.load(handle)
	handle.close()
	return [W1, W2]


# .................................................................................
# ... Save the current results to an output file. Useful when computation is taking a long time.
# .................................................................................

def save_model(W1, W2):
	handle = open("/home/liyixi/hw2/saved_W1_cluster.data", "wb+")
	np.save(handle, W1, allow_pickle = False)
	handle.close()

	handle = open("/home/liyixi/hw2/saved_W2_cluster.data", "wb+")
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
		# filename = sys.argv[1]  # feel free to read the file in a different way

		# ... load in the file, tokenize it and assign each token an index.
		# ... the full sequence of characters is encoded in terms of their one-hot positions
		fullsequence = pickle.load(open("/home/liyixi/hw2/w2v_fullrec.p", "rb"))
		wordcodes = pickle.load(open("/home/liyixi/hw2/w2v_wordcodes.p", "rb"))
		uniqueWords = pickle.load(open("/home/liyixi/hw2/w2v_uniqueWords.p", "rb"))
		wordcounts = pickle.load(open("/home/liyixi/hw2/w2v_wordcounts.p", "rb"))
		print("Full sequence loaded...")
		print("Total unique words: ", len(uniqueWords))
		samplingTable = pickle.load(open("/home/liyixi/hw2/w2v_cumulative_dict.p", "rb"))

		# ... we've got the word indices and the sampling table. Begin the training.
		# ... NOTE: If you have already trained a model earlier, preload the results (set preload=True) (This would save you a lot of unnecessary time)
		# ... If you just want to load an earlier model and NOT perform further training, comment out the train_vectors() line
		# ... ... and uncomment the load_model() line

		train_vectors(preload = False)
		print(np.asarray(word_embeddings).shape)
		print(np.asarray(proj_embeddings).shape)

	else:
		print("Please provide a valid input filename")
		sys.exit()

# python -m cProfile -s time myscript.py
