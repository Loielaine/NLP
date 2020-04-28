"""
Optimized Latent Dirichlet Allocation (LDA) <https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation> in Python.

For a faster implementation of LDA (parallelized for multicore machines), see also gensim.models.ldamulticore.

This module allows both LDA model estimation from a training corpus and inference of topic distribution on new, unseen documents. The model can also be updated with new documents for online training.

The core estimation code is based on the onlineldavb.py script, by Hoffman, Blei, Bach: Online Learning for Latent Dirichlet Allocation, NIPS 2010.

The algorithm:

Is streamed: training documents may come in sequentially, no random access required.

Runs in constant memory w.r.t. the number of documents: size of the training corpus does not affect memory footprint, can process corpora larger than RAM.

Is distributed: makes use of a cluster of machines, if available, to speed up model estimation.
"""
import argparse
from glob import glob

from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.wrappers import LdaMallet
from nltk.tokenize import TreebankWordTokenizer

kTOKENIZER = TreebankWordTokenizer()
kDOC_NORMALIZER = True
import time
from lda import VocabBuilder, tokenize_file
import logging

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)


def report_topics(outputfilename, topics, limit = 50):
	topicsfile = open(outputfilename + ".topics", 'w')
	for i in range(0, len(topics)):
		topicsfile.write("------------\nTopic %i\n------------\n" % \
		                 i)
		sorted_words = sorted(topics[i][1], key = lambda x: x[1], reverse = True)
		word = 0
		for ww, prob in sorted_words:
			topicsfile.write('%0.5f\t%s\n' % (prob, ww))
			word += 1
			if word > limit:
				break


if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--doc_dir", help = "Where we read the source documents",
	                       type = str, default = "", required = False)
	argparser.add_argument("--language", help = "The language we use",
	                       type = str, default = "english", required = False)
	argparser.add_argument("--output", help = "Where we write results",
	                       type = str, default = "./output/gensim", required = False)
	argparser.add_argument("--vocab_size", help = "Size of vocabulary",
	                       type = int, default = 1000, required = False)
	argparser.add_argument("--num_topics", help = "Number of topics",
	                       type = int, default = 5, required = False)
	argparser.add_argument("--num_iterations", help = "Number of iterations",
	                       type = int, default = 1000, required = False)
	args = argparser.parse_args()

	# Create an VocabBuilder instance
	vocab_scanner = VocabBuilder(args.language)

	# Create a list of the docs
	search_path = "./data/wiki//*.txt"
	docs = glob(search_path)
	assert len(docs) > 0, "Did not find any input docs in %s" % search_path
	# print(docs)
	# Create the vocabulary
	for ii in docs:
		vocab_scanner.scan(tokenize_file(ii))

	# Initialize the documents
	# Return a list of the top words sorted by frequency
	vocab = vocab_scanner.vocab(args.vocab_size)
	print((len(vocab), vocab[:10]))

	token_doc_list = []
	for ii in docs:
		token_doc = [x for x in tokenize_file(ii) if x in vocab]
		# print(token_doc)
		token_doc_list.append(token_doc)

	# Create a dictionary representation of the documents.
	dictionary = Dictionary(token_doc_list)
	# print(dictionary)
	# Bag-of-words representation of the documents.
	corpus = [dictionary.doc2bow(token_doc) for token_doc in token_doc_list]
	# print(corpus)
	start1 = time.time()
	lda_model = LdaModel(
		corpus = corpus,
		id2word = dictionary,
		alpha = 0.1,
		eta = 'auto',
		iterations = args.num_iterations,
		num_topics = args.num_topics
	)
	total1 = time.time() - start1
	with open(args.output + "_lda" + ".times", 'w') as out:
		out.write('time: %f s' % float(total1))
	topics1 = lda_model.show_topics(num_topics = args.num_topics, num_words = 50, log = True, formatted = False)
	report_topics(args.output + "_lda", topics1, limit = 50)

	start2 = time.time()
	lda_mallet_model = LdaMallet(
		'./Mallet/bin/mallet',
		corpus = corpus,
		id2word = dictionary,
		alpha = 0.1,
		iterations = args.num_iterations,
		num_topics = args.num_topics
	)
	total2 = time.time() - start2
	with open(args.output + "_mallet" + ".times", 'w') as out:
		out.write('time: %f s' % float(total2))
	topics2 = lda_mallet_model.show_topics(num_topics = args.num_topics, num_words = 50, log = True, formatted = False)
	report_topics(args.output + "_mallet", topics2, limit = 50)
