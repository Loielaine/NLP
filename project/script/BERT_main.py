# from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pickle
import random
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertForSequenceClassification, AdamW,get_linear_schedule_with_warmup


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
	"""Truncates a sequence pair in place to the maximum length."""

	# This is a simple heuristic which will always truncate the longer sequence
	# one token at a time. This makes more sense than truncating an equal percent
	# of tokens from each, since if one sequence is very short then each token
	# that's truncated likely contains more information than a longer sequence.
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_length:
			break
		if len(tokens_a) > len(tokens_b):
			tokens_a.pop()
		else:
			tokens_b.pop()


class InputFeatures(object):
	"""A single set of features of data."""

	def __init__(self, unique_id, input_ids, input_mask, segment_ids, label_id):
		self.unique_id = unique_id
		self.input_ids = input_ids
		self.input_mask = input_mask
		self.segment_ids = segment_ids
		self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
	"""Loads a data file into a list of `InputBatch`s."""

	label_map = {label: i for i, label in enumerate(label_list)}

	features = []
	for (ex_index, example) in enumerate(examples):
		if ex_index % 10000 == 0:
			logger.info("Writing example %d of %d" % (ex_index, len(examples)))

		tokens_a = tokenizer.tokenize(example.text_a)

		tokens_b = None
		if example.text_b:
			tokens_b = tokenizer.tokenize(example.text_b)
			# Modifies `tokens_a` and `tokens_b` in place so that the total
			# length is less than the specified length.
			# Account for [CLS], [SEP], [SEP] with "- 3"
			_truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
		else:
			# Account for [CLS] and [SEP] with "- 2"
			if len(tokens_a) > max_seq_length - 2:
				tokens_a = tokens_a[:(max_seq_length - 2)]

		# The convention in BERT is:
		# (a) For sequence pairs:
		#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
		#  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
		# (b) For single sequences:
		#  tokens:   [CLS] the dog is hairy . [SEP]
		#  type_ids: 0   0   0   0  0     0 0
		#
		# Where "type_ids" are used to indicate whether this is the first
		# sequence or the second sequence. The embedding vectors for `type=0` and
		# `type=1` were learned during pre-training and are added to the wordpiece
		# embedding vector (and position vector). This is not *strictly* necessary
		# since the [SEP] token unambiguously separates the sequences, but it makes
		# it easier for the model to learn the concept of sequences.
		#
		# For classification tasks, the first vector (corresponding to [CLS]) is
		# used as as the "sentence vector". Note that this only makes sense because
		# the entire model is fine-tuned.
		tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
		segment_ids = [0] * len(tokens)

		if tokens_b:
			tokens += tokens_b + ["[SEP]"]
			segment_ids += [1] * (len(tokens_b) + 1)

		input_ids = tokenizer.convert_tokens_to_ids(tokens)

		# The mask has 1 for real tokens and 0 for padding tokens. Only real
		# tokens are attended to.
		input_mask = [1] * len(input_ids)

		# Zero-pad up to the sequence length.
		padding = [0] * (max_seq_length - len(input_ids))
		input_ids += padding
		input_mask += padding
		segment_ids += padding

		assert len(input_ids) == max_seq_length
		assert len(input_mask) == max_seq_length
		assert len(segment_ids) == max_seq_length

		if output_mode == "classification":
			label_id = label_map[example.label]
		elif output_mode == "regression":
			label_id = float(example.label)
		else:
			raise KeyError(output_mode)

		if ex_index < 5:
			logger.info("*** Example ***")
			logger.info("unique_id: %s" % (example.unique_id))
			logger.info("tokens: %s" % " ".join(
				[str(x) for x in tokens]))
			logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
			logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
			logger.info(
				"segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
			logger.info("label: %s (id = %d)" % (example.label, label_id))

		features.append(
			InputFeatures(unique_id = ex_index,
						  input_ids = input_ids,
			              input_mask = input_mask,
			              segment_ids = segment_ids,
			              label_id = label_id))
	return features


def convert_features_to_tensor(features):
	all_input_ids = torch.tensor([f.input_ids for f in features], dtype = torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in features], dtype = torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype = torch.long)
	all_label_ids = torch.tensor([f.label_id for f in features], dtype = torch.long)
	tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
	return tensor_data

class InputExample(object):
	def __init__(self, unique_id, text_a, text_b, label):
		self.unique_id = unique_id
		self.text_a = text_a
		self.text_b = text_b
		self.label = label

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

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# Required parameters
	parser.add_argument("--data_dir",
	                    default = '/home/liyixi/project/data/',
	                    type = str)
	parser.add_argument("--task_name",
	                    default = None,
	                    type = str,
	                    required = True,
	                    help = "The name of the task to train.")
	parser.add_argument("--bert_model", default = 'bert-base-uncased', type = str,
	                    help = "Bert pre-trained model selected in the list: bert-base-uncased, "
	                           "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
	                           "bert-base-multilingual-cased, bert-base-chinese.")
	parser.add_argument("--output_dir",
	                    default = '/home/liyixi//project/output/',
	                    type = str,
	                    help = "The output directory where the model predictions and checkpoints will be written.")

	# Other Parameters
	parser.add_argument("--cache_dir",
	                    default = "",
	                    type = str,
	                    help = "Where do you want to store the pre-trained models downloaded from s3")
	parser.add_argument("--max_seq_length",
	                    default = 128,
	                    type = int,
	                    help = "The maximum total input sequence length after WordPiece tokenization. \n"
	                           "Sequences longer than this will be truncated, and sequences shorter \n"
	                           "than this will be padded.")
	parser.add_argument("--do_train",
	                    action = 'store_true',
	                    help = "Whether to run training.")
	parser.add_argument("--do_eval",
	                    action = 'store_true',
	                    help = "Whether to run eval on the dev set.")
	parser.add_argument("--do_test",
	                    action = 'store_true',
	                    help = "Whether to run eval on the test set.")
	parser.add_argument("--do_lower_case",
	                    action = 'store_true',
	                    help = "Set this flag if you are using an uncased model.")
	parser.add_argument("--train_batch_size",
	                    default = 32,
	                    type = int,
	                    help = "Total batch size for training.")
	parser.add_argument("--eval_batch_size",
	                    default = 8,
	                    type = int,
	                    help = "Total batch size for eval.")
	parser.add_argument("--test_batch_size",
	                    default = 8,
	                    type = int,
	                    help = "Total batch size for test.")
	parser.add_argument("--warmup_proportion",
	                    default = 0.1,
	                    type = float,
	                    help = "Proportion of training to perform linear learning rate warmup for. "
	                           "E.g., 0.1 = 10% of training.")
	parser.add_argument("--learning_rate",
	                    default = 5e-5,
	                    type = float,
	                    help = "The initial learning rate for Adam.")
	parser.add_argument("--num_train_epochs",
	                    default = 3.0,
	                    type = float,
	                    help = "Total number of training epochs to perform.")
	parser.add_argument("--no_cuda",
	                    action = 'store_true',
	                    help = "Whether not to use CUDA when available")
	parser.add_argument('--overwrite_output_dir',
	                    action = 'store_true',
	                    help = "Overwrite the content of the output directory")
	parser.add_argument("--local_rank",
	                    type = int,
	                    default = -1,
	                    help = "local_rank for distributed training on gpus")
	parser.add_argument('--seed',
	                    type = int,
	                    default = 42,
	                    help = "random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps',
	                    type = int,
	                    default = 1,
	                    help = "Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--print_every_iters",
	                    help = "How often to print " + "updates during training",
	                    type = int,
	                    default = 1,
	                    required = False)
	parser.add_argument(
		"--load_model_file",
		help = "Load the specified " + "saved model for testing",
		type = str,
		default = None)

	args = parser.parse_args()
	# args = parser.parse_args(['--task_name=four_sentences_pretrain', '--bert_model=/Users/Loielaine/Desktop/umich-2020/SI630/project/pretraining/pretraining-len64-lr5e-05','--do_train', '--do_eval', '--do_test'])
	# args = parser.parse_args(['--task_name=four_sentences_pretrain_test', '--do_test',
	#                           '--load_model_file=/Users/Loielaine/Desktop/umich-2020/SI630/project/output/four_sentences/bert-base-uncased-four_sentences-epoch-2.mdl'])
	# # Set device
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		n_gpu = torch.cuda.device_count()
	else:
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		n_gpu = 1
	args.device = device

	# Set logging
	logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
	                    datefmt = '%m/%d/%Y %H:%M:%S',
	                    level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
	logger = logging.getLogger(__name__)
	logger.info("device: {} n_gpu: {}, distributed training: {}".format(
		device, n_gpu, bool(args.local_rank != -1)))

	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	task_name = args.task_name
	label_list = [0, 1]
	num_labels = len(label_list)
	output_mode = "classification"
	print_every_iters = args.print_every_iters

	if args.do_train and args.do_eval:
		# Set parameters
		args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

		tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case = args.do_lower_case)
		model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels = num_labels)
		device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		model.to(device)
		print("Initiate model -- Done!")

		global_step = 0
		nb_tr_steps = 0
		tr_loss = 0
		# Load data
		train_examples = load_data(args.data_dir, 'train')
		print("Load train examples -- Done!")
		# print("length of training examples:" , len(train_examples))

		# print("length of training examples:" , len(train_examples))

		cached_train_features_file = os.path.join(args.data_dir, 'train_{0}_{1}_{2}'.format(
			list(filter(None, args.bert_model.split('/'))).pop(),
			str(args.max_seq_length),
			str(task_name)))
		try:
			with open(cached_train_features_file, "rb") as reader:
				train_features = pickle.load(reader)
		except:
			train_features = convert_examples_to_features(
				train_examples, label_list, args.max_seq_length, tokenizer, output_mode)
			if args.local_rank == -1:
				logger.info("  Saving train features into cached file %s", cached_train_features_file)
				with open(cached_train_features_file, "wb") as writer:
					pickle.dump(train_features, writer)

		train_tensor = convert_features_to_tensor(train_features)
		print("Convert features to tensor -- Done!")
		train_dataloader = DataLoader(train_tensor, sampler = RandomSampler(train_tensor),
		                              batch_size = args.train_batch_size)
		print("Load data -- Done!")

		# Load data
		val_examples = load_data(args.data_dir, 'eval')
		print("Load val examples -- Done!")


		cached_val_features_file = os.path.join(args.data_dir, 'val_{0}_{1}_{2}'.format(
			list(filter(None, args.bert_model.split('/'))).pop(),
			str(args.max_seq_length),
			str(task_name)))
		try:
			with open(cached_val_features_file, "rb") as reader:
				val_features = pickle.load(reader)
		except:
			val_features = convert_examples_to_features(
				val_examples, label_list, args.max_seq_length, tokenizer, output_mode)
			if args.local_rank == -1:
				logger.info("  Saving val features into cached file %s", cached_val_features_file)
				with open(cached_val_features_file, "wb") as writer:
					pickle.dump(val_features, writer)

		val_tensor = convert_features_to_tensor(val_features)
		print("Convert features to tensor -- Done!")
		val_dataloader = DataLoader(val_tensor, sampler = SequentialSampler(val_tensor),
		                            batch_size = args.eval_batch_size)
		print("Load data -- Done!")

		num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

		# Set optimizer as AdamW
		optimizer = AdamW(model.parameters(),
			                  lr = args.learning_rate,
			                  eps = 1e-8)


		# Create the learing rate schedular
		scheduler = get_linear_schedule_with_warmup(
				optimizer,
				num_warmup_steps = int(0.1 * num_train_optimization_steps),
				num_training_steps = num_train_optimization_steps
			)

		logger.info("***** Running training *****")
		logger.info("  Num examples = %d", len(train_examples))
		logger.info("  Batch size = %d", args.train_batch_size)
		logger.info("  Num steps = %d", num_train_optimization_steps)
		model.train()
		loss_list = defaultdict(list)
		acc_list = defaultdict(list)
		logger.info("***** Train outputs *****")
		for epoch in trange(int(args.num_train_epochs), desc = "Epoch", disable = args.local_rank not in [-1, 0]):
			logger.info("epoch: %d", epoch)
			tr_loss = 0
			nb_tr_examples, nb_tr_steps = 0, 0
			for step, batch in enumerate(
					tqdm(train_dataloader, desc = "Iteration", disable = args.local_rank not in [-1, 0])):
				batch = tuple(t.to(device) for t in batch)
				input_ids, input_mask, segment_ids, label_ids = batch
				# define a new function to compute loss values for both output_modes
				loss, logits = model(input_ids, token_type_ids = segment_ids, attention_mask = input_mask,
				                     labels = label_ids)
				# loss_fct = nn.CrossEntropyLoss()
				# loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
				# print(logits.view(-1, num_labels))
				# print(logits.argmax(1))

				tr_loss += loss.item()
				nb_tr_examples += input_ids.size(0)
				nb_tr_steps += 1

				if (step + 1) % args.gradient_accumulation_steps == 0:
					optimizer.step()
					scheduler.step()
					optimizer.zero_grad()
					global_step += 1
					# tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
					# tb_writer.add_scalar('loss', loss.item(), global_step)
					if step % print_every_iters == 0:
						avg_loss = float(tr_loss / nb_tr_examples)
						loss_list[epoch].append(avg_loss)
						acc = int((logits.argmax(1) == label_ids).sum()) / len(label_ids)
						acc_list[epoch].append(acc)
						logger.info('Epoch: %d [%d], loss: %1.3f, acc: %1.3f' \
						            % (epoch, step, avg_loss, acc))

			# save model for each epochs
			output_model_file = os.path.join(args.output_dir,
			                                 '%s-%s-epoch-%d.mdl' % (args.bert_model, task_name, epoch))
			torch.save(model, output_model_file)

			model.eval()
			eval_loss = 0
			nb_eval_steps = 0
			preds = []
			out_label_ids = None
			for input_ids, input_mask, segment_ids, label_ids in tqdm(val_dataloader, desc = "Evaluating"):
				input_ids = input_ids.to(device)
				input_mask = input_mask.to(device)
				segment_ids = segment_ids.to(device)
				label_ids = label_ids.to(device)
				tmp_eval_loss = []
				with torch.no_grad():
					loss, logits = model(input_ids, token_type_ids = segment_ids, attention_mask = input_mask,
					                     labels = label_ids)
					tmp_eval_loss.append(loss.item())
				# loss_fct = nn.CrossEntropyLoss()
				# tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

				# print(logits.view(-1, num_labels))

				eval_loss += sum(tmp_eval_loss) / len(tmp_eval_loss)
				nb_eval_steps += 1
				if len(preds) == 0:
					preds.append(logits.detach().cpu().numpy())
					out_label_ids = label_ids.detach().cpu().numpy()
				else:
					preds[0] = np.append(
						preds[0], logits.detach().cpu().numpy(), axis = 0)
					out_label_ids = np.append(
						out_label_ids, label_ids.detach().cpu().numpy(), axis = 0)

			eval_loss = eval_loss / nb_eval_steps
			preds = preds[0]
			preds = preds.argmax(1)
			# print("eval_prediction:", preds)
			# print("eval_label:", out_label_ids)

			output_eval_pred_file = os.path.join(args.output_dir, '%s-%s-eval_pred.txt' % (args.bert_model, task_name))
			with open(output_eval_pred_file, "w") as writer:
				logger.info("***** Eval predictions *****")
				writer.write("Id,Prediction,Label\n")
				for i in range(len(preds)):
					writer.write("%d,%d,%d\n" % (i, preds[i], out_label_ids[i]))

			acc = (preds == out_label_ids).mean()
			loss = tr_loss / global_step if args.do_train else None
			result = {}
			result['eval_loss'] = eval_loss
			result['global_step'] = global_step
			result['loss'] = loss
			result['accuracy'] = acc

			output_eval_file = os.path.join(args.output_dir, '%s-%s-eval_outputs.txt' % (args.bert_model, task_name))
			with open(output_eval_file, "w") as writer:
				logger.info("***** Eval results *****")
				for key in sorted(result.keys()):
					logger.info("  %s = %s", key, str(result[key]))
					writer.write("%s = %s\n" % (key, str(result[key])))

		# save train outputs
		output_train_file = os.path.join(args.output_dir, '%s-%s-train_outputs.txt' % (args.bert_model, task_name))
		with open(output_train_file, "w") as writer:
			for i in range(int(args.num_train_epochs)):
				writer.write('Epoch: %d\n' % i)
				for step in range(len(loss_list[i])):
					writer.write("step%d: loss: %1.3f, acc: %1.3f" \
					             % (step, loss_list[i][step], acc_list[i][step]))
					writer.write('\n')
				writer.write('\n')
		# save config
		model_to_save = model.module if hasattr(model, 'module') else model

	if args.do_test:
		# load model
		if args.do_train:
			load_model_file = os.path.join(args.output_dir,
			                               '%s-%s-epoch-%d.mdl' % (
				                               args.bert_model, task_name, int(args.num_train_epochs) - 1))
			model = torch.load(load_model_file)
		elif args.load_model_file is not None:
			model = torch.load(args.load_model_file)
			print("load model %s" % args.load_model_file)
		else:
			model = BertForSequenceClassification.from_pretrained(args.bert_model, num_labels = num_labels)

		tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case = args.do_lower_case)
		device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		model.to(device)
		# Load data
		test_examples = load_data(args.data_dir, 'test')
		print("Load test examples -- Done!")


		cached_test_features_file = os.path.join(args.data_dir, 'test_{0}_{1}_{2}'.format(
			list(filter(None, args.bert_model.split('/'))).pop(),
			str(args.max_seq_length),
			str(args.task_name)))
		try:
			with open(cached_test_features_file, "rb") as reader:
				test_features = pickle.load(reader)
		except:
			test_features = convert_examples_to_features(
				test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
			if args.local_rank == -1:
				logger.info("  Saving test features into cached file %s", cached_test_features_file)
				with open(cached_test_features_file, "wb") as writer:
					pickle.dump(test_features, writer)

		test_tensor = convert_features_to_tensor(test_features)
		print("Convert features to tensor -- Done!")
		test_dataloader = DataLoader(test_tensor, sampler = SequentialSampler(test_tensor),
		                             batch_size = args.test_batch_size)
		print("Load data -- Done!")

		# Tracking variables
		predictions, true_labels = [], []
		# Predict
		for batch in test_dataloader:
			# Add batch to GPU
			batch = tuple(t.to(device) for t in batch)

			# Unpack the inputs from our dataloader
			input_ids, input_mask, segment_ids, labels = batch

			# Telling the model not to compute or store gradients, saving memory and
			# speeding up prediction
			with torch.no_grad():
				# Forward pass, calculate logit predictions
				outputs = model(input_ids, token_type_ids = segment_ids,
				                attention_mask = input_mask)

			logits = outputs[0]
			print(logits)
			# Move logits and labels to CPU
			logits = logits.detach().cpu().numpy()
			label_ids = labels.to('cpu').numpy()

			# Store predictions and true labels
			predictions.append(logits)
			true_labels.append(label_ids)

		print('Prediction -- Done!')
		# Combine the results across all batches.
		flat_predictions = np.concatenate(predictions, axis = 0)

		# For each sample, pick the label (0 or 1) with the higher score.
		flat_predictions = np.argmax(flat_predictions, axis = 1).flatten()

		# Combine the correct labels for each batch into a single list.
		flat_true_labels = np.concatenate(true_labels, axis = 0)

		acc = (flat_predictions == flat_true_labels).mean()
		print(acc)
		preds_list = list(flat_predictions)
		# save train outputs
		output_test_file = os.path.join(args.output_dir, '%s-%s-test_preds.txt' % (args.bert_model, task_name))
		with open(output_test_file, "w") as writer:
			writer.write("Id,Pred,Label,Text_a,Text_b")
			for i in range(len(test_examples)):
				writer.write('%d,%d,%d,%s,%s' %(test_examples[i].unique_id,preds_list[i],test_examples[i].label,test_examples[i].text_a,test_examples[i].text_b))
				writer.write('\n')
		writer.close()
		matthews_set = []

		# Evaluate each test batch using Matthew's correlation coefficient
		print('Calculating Matthews Corr. Coef. for each batch...')

		# For each input batch...
		for i in range(len(true_labels)):
			# The predictions for this batch are a 2-column ndarray (one column for "0"
			# and one column for "1"). Pick the label with the highest value and turn this
			# in to a list of 0s and 1s.
			pred_labels_i = np.argmax(predictions[i], axis = 1).flatten()

			# Calculate and store the coef for this batch.
			matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
			matthews_set.append(matthews)

		# Calculate the MCC
		mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
		print(mcc)

