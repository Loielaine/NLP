import pandas as pd
import torch
from torch import nn

class InputExample(object):
	def __init__(self, unique_id, text_a, text_b, label):
		self.unique_id = unique_id
		self.text_a = text_a
		self.text_b = text_b
		self.label = label

# four_sentences
# def get_label_data(df):
# 	df['text_a'] = df[['InputSentence1', 'InputSentence2', 'InputSentence3', 'InputSentence4']].agg(' '.join,
# 	                                                                                                axis = 1)
# 	Ending1 = df[['text_a', 'RandomFifthSentenceQuiz1', 'AnswerRightEnding']].copy(deep = True)
# 	Ending1.loc[Ending1['AnswerRightEnding'] == 2, 'label'] = 0
# 	Ending1.loc[Ending1['AnswerRightEnding'] == 1, 'label'] = 1
# 	Ending1.columns = ['text_a', 'text_b', 'AnswerRightEnding', 'label']
# 	Ending2 = df[['text_a', 'RandomFifthSentenceQuiz2', 'AnswerRightEnding']].copy(deep = True)
# 	Ending2.loc[Ending2['AnswerRightEnding'] == 1, 'label'] = 0
# 	Ending2.loc[Ending2['AnswerRightEnding'] == 2, 'label'] = 1
# 	Ending2.columns = ['text_a', 'text_b', 'AnswerRightEnding', 'label']
# 	label_df = Ending1.append(Ending2, ignore_index = True)
# 	examples = []
# 	for i in range(len(label_df)):
# 		examples.append(
# 			InputExample(unique_id = i, text_a = label_df['text_a'].iloc[i], text_b = label_df['text_b'].iloc[i],
# 			             label = label_df['label'].iloc[i]))
# 	return examples

# four_sentences_reverse
# def get_label_data(df):
# 	df['text_a'] = df[['InputSentence4', 'InputSentence3', 'InputSentence2', 'InputSentence1']].agg(' '.join,
# 	                                                                                                axis = 1)
# 	Ending1 = df[['text_a', 'RandomFifthSentenceQuiz1', 'AnswerRightEnding']].copy(deep = True)
# 	Ending1.loc[Ending1['AnswerRightEnding'] == 2, 'label'] = 0
# 	Ending1.loc[Ending1['AnswerRightEnding'] == 1, 'label'] = 1
# 	Ending1.columns = ['text_a', 'text_b', 'AnswerRightEnding', 'label']
# 	Ending2 = df[['text_a', 'RandomFifthSentenceQuiz2', 'AnswerRightEnding']].copy(deep = True)
# 	Ending2.loc[Ending2['AnswerRightEnding'] == 1, 'label'] = 0
# 	Ending2.loc[Ending2['AnswerRightEnding'] == 2, 'label'] = 1
# 	Ending2.columns = ['text_a', 'text_b', 'AnswerRightEnding', 'label']
# 	label_df = Ending1.append(Ending2, ignore_index = True)
# 	examples = []
# 	for i in range(len(label_df)):
# 		examples.append(
# 			InputExample(unique_id = i, text_a = label_df['text_a'].iloc[i], text_b = label_df['text_b'].iloc[i],
# 			             label = label_df['label'].iloc[i]))
# 	return examples

# three_sentences
# def get_label_data(df):
# 	df['text_a'] = df[['InputSentence2', 'InputSentence3', 'InputSentence4']].agg(' '.join,
# 	                                                                                                axis = 1)
# 	Ending1 = df[['text_a', 'RandomFifthSentenceQuiz1', 'AnswerRightEnding']].copy(deep = True)
# 	Ending1.loc[Ending1['AnswerRightEnding'] == 2, 'label'] = 0
# 	Ending1.loc[Ending1['AnswerRightEnding'] == 1, 'label'] = 1
# 	Ending1.columns = ['text_a', 'text_b', 'AnswerRightEnding', 'label']
# 	Ending2 = df[['text_a', 'RandomFifthSentenceQuiz2', 'AnswerRightEnding']].copy(deep = True)
# 	Ending2.loc[Ending2['AnswerRightEnding'] == 1, 'label'] = 0
# 	Ending2.loc[Ending2['AnswerRightEnding'] == 2, 'label'] = 1
# 	Ending2.columns = ['text_a', 'text_b', 'AnswerRightEnding', 'label']
# 	label_df = Ending1.append(Ending2, ignore_index = True)
# 	examples = []
# 	for i in range(len(label_df)):
# 		examples.append(
# 			InputExample(unique_id = i, text_a = label_df['text_a'].iloc[i], text_b = label_df['text_b'].iloc[i],
# 			             label = label_df['label'].iloc[i]))
# 	return examples

def get_label_data(df):
	df['text_a'] = df[['InputSentence1', 'InputSentence2','InputSentence3', 'InputSentence4']].agg(' '.join,
	                                                                                                axis = 1)
	Ending1 = df[['text_a', 'RandomFifthSentenceQuiz1', 'AnswerRightEnding']].copy(deep = True)
	Ending1.loc[Ending1['AnswerRightEnding'] == 2, 'label'] = 0
	Ending1.loc[Ending1['AnswerRightEnding'] == 1, 'label'] = 1
	Ending1.columns = ['text_a', 'text_b', 'AnswerRightEnding', 'label']
	Ending2 = df[['text_a', 'RandomFifthSentenceQuiz2', 'AnswerRightEnding']].copy(deep = True)
	Ending2.loc[Ending2['AnswerRightEnding'] == 1, 'label'] = 0
	Ending2.loc[Ending2['AnswerRightEnding'] == 2, 'label'] = 1
	Ending2.columns = ['text_a', 'text_b', 'AnswerRightEnding', 'label']
	label_df = Ending1.append(Ending2, ignore_index = True)
	print("true label proportions: ", float(sum(label_df['label'])/len(label_df)))
	examples = []
	for i in range(len(label_df)):
		examples.append(
			InputExample(unique_id = i, text_a = label_df['text_a'].iloc[i], text_b = label_df['text_b'].iloc[i],
			             label = label_df['label'].iloc[i]))
	return examples


def load_data(file_dir, action):
	if action == 'train':
		train = pd.read_csv(file_dir + '2016-val.csv')
		train_examples = get_label_data(train)
		print('Number of training sample: {:,}\n'.format(train_examples[-1].unique_id))
		return train_examples

	if action == 'eval':
		val = pd.read_csv(file_dir + '2016-test.csv')
		val_examples = get_label_data(val)
		print('Number of validate sample: {:,}\n'.format(val_examples[-1].unique_id))
		return val_examples

	if action == 'test':
		test = pd.read_csv(file_dir + '2018-val.csv')
		test_examples = get_label_data(test)
		print('Number of validate sample: {:,}\n'.format(test_examples[-1].unique_id))
		return test_examples

def get_pretraining_label_data(df):
	df['text_a'] = df[['sentence1', 'sentence2','sentence3', 'sentence4']].agg(' '.join,
	                                                                                                axis = 1)
	df['text_b'] = df[['sentence5']]
	examples = []
	for i in range(len(df)):
		examples.append(
			InputExample(unique_id = i, text_a = df['text_a'].iloc[i], text_b = df['text_b'].iloc[i],
			             label = 1))
	return examples


def load_pretraining_data(file_dir):
	train = pd.read_csv(file_dir + 'ROCStories2016.csv')
	train_examples = get_pretraining_label_data(train)
	val = pd.read_csv(file_dir + 'ROCStories2017.csv')
	val_examples = get_pretraining_label_data(val)
	return train_examples, val_examples


# def load_embeddings(vocab_size,embedding_dim , emb_type = 'new', emb_file_name = None):
# 	if emb_type == 'new':
# 		print('Creating new trainable embeddings')
# 		word_embeddings = nn.Embedding(vocab_size,
# 		                               embedding_dim)
# 	elif emb_type == 'twitter':
#
# 		pass
# 	elif emb_type == 'wiki' or emb_type == 'wikipedia':
#
# 		pass
# 	else:
# 		raise Error('unknown embedding type!: "%s"' % emb_type)
#
# 	return word_embeddings
