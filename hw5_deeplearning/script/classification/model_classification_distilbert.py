import argparse
import datetime
import os
import pickle
import time
import logging
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm, trange
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW,get_linear_schedule_with_warmup


def read_data(data_path, tokenizer, max_seq_length):
	"""
	Read in data from txt file and return TensorDataset with
	input_ids, attention_masks, labels
	"""

	# Read data in dataframe format
	with open(data_path, encoding = 'utf-8') as f:
		data = {}
		headers = f.readline().strip().split("\t")
		for h in headers:
			data[h] = []
		line = f.readline()
		while line:
			pos = line.find("\t")
			data[headers[0]].append(line[:pos])
			data[headers[1]].append(line[pos + 1:-1])
			line = f.readline()
	dt = pd.DataFrame(data)
	# dt = pd.read_csv(data_path, sep = "\t")
	# dt_lines = open(data_path).readlines()

	# Convert text and labels in lists
	text = dt['text'].values
	class_label = dt['class'].values
	labels = []
	for label in class_label:
		if label == 'machine':
			labels.append(1)
		else:
			labels.append(0)

	input_ids = []
	attention_masks = []

	# For every sentence in data text
	for sent in text:
		encoded_dict = tokenizer.encode_plus(
			sent,
			add_special_token = True,
			max_length = max_seq_length,
			pad_to_max_length = True,
			return_attention_mask = True,
			return_tensors = 'pt'
		)

		# Add the encoded sentence to the list.
		input_ids.append(encoded_dict['input_ids'])
		# And its attention mask (simply differentiates padding from non-padding).
		attention_masks.append(encoded_dict['attention_mask'])

	# Convert input_ids, attention_masks and label into tensors
	input_ids = torch.cat(input_ids, dim = 0)
	attention_masks = torch.cat(attention_masks, dim = 0)
	labels = torch.tensor(labels)
	# print(labels.size())

	# Convert dataset into TensorDataset format
	dataset = TensorDataset(input_ids, attention_masks, labels)

	return dataset


def read_test_data(data_path, tokenizer, max_seq_length):
	"""
	Read in data from txt file and return TensorDataset with
	input_ids, attention_masks, labels
	"""

	# Read data in dataframe format
	with open(data_path, encoding = 'utf-8') as f:
		data = {}
		headers = f.readline().strip().split("\t")
		for h in headers:
			data[h] = []
		line = f.readline()
		while line:
			pos = line.find("\t")
			data[headers[0]].append(line[:pos])
			data[headers[1]].append(line[pos + 1:-1])
			line = f.readline()
	dt = pd.DataFrame(data)
	# dt = pd.read_csv(data_path, sep = "\t",encoding="utf-8")

	# dt_lines = open(data_path).readlines()

	# Convert text and labels in lists
	text = dt['Text'].values
	# print(len(text))
	input_ids = []
	attention_masks = []

	# For every sentence in data text
	for sent in text:
		encoded_dict = tokenizer.encode_plus(
			sent,
			add_special_token = True,
			max_length = max_seq_length,
			pad_to_max_length = True,
			return_attention_mask = True,
			return_tensors = 'pt'
		)

		# Add the encoded sentence to the list.
		input_ids.append(encoded_dict['input_ids'])
		# And its attention mask (simply differentiates padding from non-padding).
		attention_masks.append(encoded_dict['attention_mask'])

	# Convert input_ids, attention_masks and label into tensors
	input_ids = torch.cat(input_ids, dim = 0)
	attention_masks = torch.cat(attention_masks, dim = 0)

	# Convert dataset into TensorDataset format
	dataset = TensorDataset(input_ids, attention_masks)

	return dataset


def flat_accuracy(preds, labels):
	"""
	Function to calculate the accuracy of our predictions vs labels
	"""
	pred_flat = np.argmax(preds, axis = 1).flatten()
	labels_flat = labels.flatten()
	return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
	"""
	Takes a time in seconds and returns a string hh:mm:ss
	"""
	# Round to the nearest second.
	elapsed_rounded = int(round((elapsed)))

	# Format as hh:mm:ss
	return str(datetime.timedelta(seconds = elapsed_rounded))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# Required parameters
	parser.add_argument("--data_dir",
	                    default = '/home/liyixi/hw5/data/classification/',
	                    type = str)
	parser.add_argument("--task_name",
	                    default = None,
	                    type = str,
	                    required = True,
	                    help = "The name of the task to train.")
	parser.add_argument("--output_dir",
	                    default = '/home/liyixi/hw5/output/classification/',
	                    type = str,
	                    help = "The output directory where the model predictions and checkpoints will be written.")
	parser.add_argument("--do_train",
	                    action = 'store_true',
	                    help = "Whether to run training.")
	parser.add_argument("--do_eval",
	                    action = 'store_true',
	                    help = "Whether to run eval on the dev set.")
	parser.add_argument("--do_test",
	                    action = 'store_true',
	                    help = "Whether to run test on the test set.")
	parser.add_argument("--do_test_no_label",
	                    action = 'store_true',
	                    help = "Whether to run test on the test set.")
	parser.add_argument("--do_test_generation",
	                    action = 'store_true',
	                    help = "Whether to test the generation.")
	parser.add_argument("--train_batch_size",
	                    default = 32,
	                    type = int,
	                    help = "Total batch size for training.")
	parser.add_argument("--eval_batch_size",
	                    default = 32,
	                    type = int,
	                    help = "Total batch size for eval.")
	parser.add_argument("--test_batch_size",
	                    default = 32,
	                    type = int,
	                    help = "Total batch size for test.")
	parser.add_argument("--learning_rate", nargs = '+',
	                    default = [],
	                    type = str,
	                    help = "The initial learning rate for optimizer.")
	parser.add_argument("--num_train_epochs",
	                    default = 5,
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
	parser.add_argument("--seed", type = int, default = 9001, help = "random seed for initialization")
	parser.add_argument(
		"--load_pretraining_model",
		help = "Load the pre-training model",
		type = str,
		default = None)
	parser.add_argument(
		"--load_model_name",
		help = "Load the specified " + "saved model for testing",
		type = str,
		default = None)
	parser.add_argument(
		"--generate_text_name",
		help = "Load the specified " + "saved model for testing",
		type = str,
		default = None)
	args = parser.parse_args()
	# Step 1 args
	# args = parser.parse_args(['--task_name=bert_classification', '--learning_rate=5e-6 5e-5 5e-4',
	#                            '--load_pretraining_model=/Users/Loielaine/Desktop/umich-2020/SI630/hw/hw5/output/fine-tuning/pretraining-len32-lr5e-05',
	#                            '--do_train', '--do_eval'])
	# args = parser.parse_args(['--task_name=bert_classification', '--learning_rate=5e-5',
	#                            '--do_train', '--do_eval'])
	# # Step 2 args
	# args = parser.parse_args(['--task_name=bert_classification_test', '--do_test',
	#                            '--load_model_name=/home/liyixi/hw5/output/classification/bert_classification-len32-lr5e-05-epoch4'])
	#  Set seed
	seed = args.seed
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	# Set device
	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		n_gpu = torch.cuda.device_count()
		torch.manual_seed(args.seed)
	else:
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		n_gpu = 1
		torch.cuda.manual_seed_all(args.seed)
	args.device = device

	# Set logging
	timestr = time.strftime("%Y%m%d-%H%M%S")
	logging.basicConfig(filename = os.path.join(args.output_dir, 'log_{0}_{1}.log'.format(
		str(args.task_name), timestr)),
	                    filemode = 'a', format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
	                    datefmt = '%m/%d/%Y %H:%M:%S',
	                    level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
	logger = logging.getLogger(__name__)
	logger.info("device: {} n_gpu: {}, distributed training: {}".format(
		device, n_gpu, bool(args.local_rank != -1)))

	if args.do_train and args.do_eval:
		# Load pre-training model
		if args.load_pretraining_model is not None:
			# Set tokenizer
			tokenizer = DistilBertTokenizer.from_pretrained(args.load_pretraining_model, do_lower_case = True)
			# tokenizer = DistilBertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
			# Set model
			model = DistilBertForSequenceClassification.from_pretrained(args.load_pretraining_model, num_labels = 2,
			                                                      output_attentions = False,
			                                                      output_hidden_states = False)
		else:
			tokenizer = DistilBertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
			model = DistilBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = 2,
			                                                      output_attentions = False,
			                                                      output_hidden_states = False)

		model.to(device)

		# Parameters for validation
		lr_list = list(map(float, args.learning_rate[0].split(' ')))
		length = 32
		eps = 1e-8

		logger.info("***** Load training data*****")
		# Read and save training input_ids
		cached_input_file = os.path.join(args.data_dir, 'train_{0}_{1}'.format(
			str(args.task_name), str(length)))
		try:
			with open(cached_input_file, "rb") as reader:
				train_inputs = pickle.load(reader)
				reader.close()
		except:
			train_inputs = read_data(args.data_dir + 'train.tsv', tokenizer, length)

			if args.local_rank == -1:
				logger.info("  Saving train features into cached file %s", cached_input_file)
				with open(cached_input_file, "wb") as writer:
					pickle.dump(train_inputs, writer)
					writer.close()

		logger.info("***** Load eval data*****")
		# Read and save training input_ids
		cached_input_file = os.path.join(args.data_dir, 'dev_{0}_{1}'.format(
			str(args.task_name), str(length)))
		try:
			with open(cached_input_file, "rb") as reader:
				eval_inputs = pickle.load(reader)
				reader.close()
		except:
			eval_inputs = read_data(args.data_dir + 'dev.tsv', tokenizer, length)

			if args.local_rank == -1:
				logger.info("  Saving eval features into cached file %s", cached_input_file)
				with open(cached_input_file, "wb") as writer:
					pickle.dump(eval_inputs, writer)
					writer.close()

		# Create DataLoader
		train_dataloader = DataLoader(
			train_inputs,
			sampler = RandomSampler(train_inputs),
			batch_size = args.train_batch_size
		)
		eval_dataloader = DataLoader(
			eval_inputs,
			sampler = SequentialSampler(eval_inputs),
			batch_size = args.eval_batch_size
		)

		for learning_rate in lr_list:
			logger.info("***** Validation parameters*****")
			logger.info(" Learning rate = %f" % learning_rate)

			# Train model
			logger.info("***** Run training and evaluating *****")
			logger.info("  Num of train examples = %d", len(train_dataloader))
			logger.info("  Train batch size = %d", args.train_batch_size)
			logger.info("  Num of eval examples = %d", len(eval_dataloader))
			logger.info("  Eval batch size = %d", args.eval_batch_size)

			# Set optimizer as AdamW
			optimizer = AdamW(model.parameters(),
			                  lr = learning_rate,
			                  eps = eps)

			# Total number of training steps is [number of batches] x [number of epochs].
			# (Note that this is not the same as the number of training samples).
			total_steps = len(train_dataloader) * args.num_train_epochs

			# Create the learing rate schedular
			scheduler = get_linear_schedule_with_warmup(
				optimizer,
				num_warmup_steps = int(0.1 * total_steps),
				num_training_steps = total_steps
			)

			# ========================================
			#               Training
			# ========================================

			# Perform one full pass over the training set.

			# Store training statistics (loss)
			training_stats = []
			# Measure the total training time
			total_t0 = time.time()
			for epoch in trange(int(args.num_train_epochs), desc = "Epoch", disable = args.local_rank not in [-1, 0]):
				logger.info("epoch: %d", epoch)
				# Reset the total loss for this epoch.
				total_train_loss = 0
				t0 = time.time()
				model.train()

				for step, batch in enumerate(
						tqdm(train_dataloader, desc = "Iteration", disable = args.local_rank not in [-1, 0])):
					b_input_ids, b_input_mask, b_labels = batch
					b_input_ids = b_input_ids.to(device)
					b_input_mask = b_input_mask.to(device)
					b_labels = b_labels.to(device)

					# Always clear any previously calculated gradients before performing a
					# backward pass. PyTorch doesn't do this automatically because
					# accumulating the gradients is "convenient while training RNNs".
					# (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
					model.zero_grad()

					# Perform a forward pass (evaluate the model on this training batch).
					# The documentation for this `model` function is here:
					# https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.DistilBertForSequenceClassification
					# It returns different numbers of parameters depending on what arguments
					# arge given and what flags are set. For our useage here, it returns
					# the loss (because we provided labels) and the "logits"--the model
					# outputs prior to activation.
					loss, logits = model(b_input_ids,
					                     
					                     attention_mask = b_input_mask,
					                     labels = b_labels)

					# Accumulate the training loss over all of the batches so that we can
					# calculate the average loss at the end. `loss` is a Tensor containing a
					# single value; the `.item()` function just returns the Python value
					# from the tensor.
					total_train_loss += loss.item()

					loss.backward()

					# Clip the norm of the gradients to 1.0.
					# This is to help prevent the "exploding gradients" problem.
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

					# Update parameters and take a step using the computed gradient.
					# The optimizer dictates the "update rule"--how the parameters are
					# modified based on their gradients, the learning rate, etc.
					optimizer.step()

					# Update the learning rate.
					scheduler.step()

					optimizer.zero_grad()

				# Calculate the average loss over all of the batches.
				avg_train_loss = total_train_loss / len(train_dataloader)

				# Measure how long this epoch took.
				training_time = format_time(time.time() - t0)
				logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))
				logger.info("  Training epcoh took: {:}".format(training_time))

				# ========================================
				#               Validation
				# ========================================
				# After the completion of each training epoch, measure our performance on
				# our validation set.

				t0 = time.time()

				# Put the model in evaluation mode--the dropout layers behave differently
				# during evaluation.
				model.eval()

				# Tracking variables
				total_eval_accuracy = 0.0
				total_eval_loss = 0.0
				nb_eval_steps = 0

				# Evaluate data for one epoch
				for batch in tqdm(eval_dataloader, desc = "Evaluating"):
					b_input_ids = batch[0].to(device)
					b_input_mask = batch[1].to(device)
					b_labels = batch[2].to(device)

					# Tell pytorch not to bother with constructing the compute graph during
					# the forward pass, since this is only needed for backprop (training).
					with torch.no_grad():
						(loss, logits) = model(b_input_ids,
						                       
						                       attention_mask = b_input_mask,
						                       labels = b_labels)

					# Accumulate the validation loss.
					total_eval_loss += loss.item()

					# Move logits and labels to CPU
					logits = logits.detach().cpu().numpy()
					label_ids = b_labels.to('cpu').numpy()

					# Calculate the accuracy for this batch of test sentences, and
					# accumulate it over all batches.
					total_eval_accuracy += flat_accuracy(logits, label_ids)

				# Report the final accuracy for this validation run.
				avg_val_accuracy = total_eval_accuracy / len(eval_dataloader)
				logger.info("  Accuracy: {0:.2f}".format(avg_val_accuracy))

				# Calculate the average loss over all of the batches.
				avg_val_loss = total_eval_loss / len(eval_dataloader)

				# Measure how long the validation run took.
				validation_time = format_time(time.time() - t0)

				logger.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
				logger.info("  Validation took: {:}".format(validation_time))

				# Record all statistics from this epoch.
				training_stats.append({
					'epoch': epoch,
					'Training Loss': avg_train_loss,
					'Valid. Loss': avg_val_loss,
					'Valid. Accuracy': avg_val_accuracy,
					'Training Time': training_time,
					'Validation Time': validation_time})

			# Save the final model to model_sav
			model_name = '%s-len%d-lr%s-epoch%d' % (
				args.task_name, length, str(learning_rate), args.num_train_epochs - 1)
			output_model_file = os.path.join(args.output_dir, model_name)
			if not os.path.exists(output_model_file):
				os.makedirs(output_model_file)

			logger.info('Saving model to %s' % output_model_file)

			# Save a trained model, configuration and tokenizer using `save_pretrained()`.
			# They can then be reloaded using `from_pretrained()`
			model_to_save = model.module if hasattr(model,
			                                        'module') else model  # Take care of distributed/parallel training
			model_to_save.save_pretrained(output_model_file)
			tokenizer.save_pretrained(output_model_file)

			# Save results to file
			output_result_file = os.path.join(args.output_dir, '%s_result.log' % args.task_name)
			with open(output_result_file, "a") as writer:
				writer.write("Sequence length: %d\n" % length)
				writer.write("Learning rate: %s\n" % str(learning_rate))
				for i in range(int(args.num_train_epochs)):
					writer.write('Epoch: %d\n' % i)
					writer.write("training loss: %1.3f, eval loss: %1.3f,eval accuracy: %1.3f" \
					             % (training_stats[i]['Training Loss'], training_stats[i]['Valid. Loss'],
					                training_stats[i]['Valid. Accuracy']))
					writer.write('\n')
				writer.write('\n')

	if args.do_test:
		# load model and tokenizer
		logger.info("***** Load test model*****")
		if args.load_model_name is not None:
			# Set tokenizer
			tokenizer = DistilBertTokenizer.from_pretrained(args.load_model_name, do_lower_case = True)
			# Set model
			model = DistilBertForSequenceClassification.from_pretrained(args.load_model_name, num_labels = 2)
			model.to(device)
			logger.info("Model loaded: %s" % args.load_model_name)
		else:
			logger.error('No model loaded!')

		learning_rate = float(args.load_model_name.split('-')[2][2:] + '-' + args.load_model_name.split('-')[3])
		length = int(args.load_model_name.split('-')[1][3:])
		logger.info(" Sequence length = %d" % length)
		logger.info(" Learning rate = %f" % learning_rate)

		logger.info("***** Load test data*****")
		# Read and save training input_ids
		cached_input_file = os.path.join(args.data_dir, 'test_{0}_{1}'.format(
			str(args.task_name), str(length)))
		try:
			with open(cached_input_file, "rb") as reader:
				test_inputs = pickle.load(reader)
				reader.close()
		except:
			test_inputs = read_data(args.data_dir + 'test.tsv', tokenizer, length)


			if args.local_rank == -1:
				logger.info("  Saving eval features into cached file %s", cached_input_file)
				with open(cached_input_file, "wb") as writer:
					pickle.dump(test_inputs, writer)
					writer.close()

		test_dataloader = DataLoader(test_inputs, sampler = SequentialSampler(test_inputs),
		                             batch_size = args.test_batch_size)

		logger.info("***** Predicting *****")
		# Tracking variables
		predictions, true_labels = [], []
		total_test_accuracy = 0.0
		# Predict
		for batch in tqdm(test_dataloader, desc = "Testing"):
			# Add batch to GPU
			batch = tuple(t.to(device) for t in batch)

			# Unpack the inputs from our dataloader
			b_input_ids, b_input_mask, b_labels = batch
			b_input_ids = b_input_ids.to(device)
			b_input_mask = b_input_mask.to(device)
			b_labels = b_labels.to(device)

			# Telling the model not to compute or store gradients, saving memory and
			# speeding up prediction
			with torch.no_grad():
				# Forward pass, calculate logit predictions
				outputs = model(b_input_ids, 
				                attention_mask = b_input_mask)

			logits = outputs[0]

			# Move logits and labels to CPU
			logits = logits.detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()

			# Calculate the accuracy for this batch of test sentences, and
			# accumulate it over all batches.
			total_test_accuracy += flat_accuracy(logits, label_ids)

			# Store predictions and true labels
			predictions.append(logits)
			true_labels.append(label_ids)

		# Combine the results across all batches.
		flat_predictions = np.concatenate(predictions, axis = 0)

		# For each sample, pick the label (0 or 1) with the higher score.
		flat_predictions = np.argmax(flat_predictions, axis = 1).flatten()

		# Combine the correct labels for each batch into a single list.
		flat_true_labels = np.concatenate(true_labels, axis = 0)

		acc = (flat_predictions == flat_true_labels).mean()
		logger.info("Testing accuracy from label comparison: %1.3f" % acc)

		# Report the final accuracy for this validation run.
		avg_test_accuracy = total_test_accuracy / len(test_dataloader)
		logger.info("Testing accuracy from average batch accuracy:  %1.3f" % avg_test_accuracy)

		# Calculate the MCC
		mcc = matthews_corrcoef(flat_true_labels, flat_predictions)
		logger.info("Testing MCC: %1.3f" % mcc)

	if args.do_test_no_label:
		# load model and tokenizer
		logger.info("***** Load test model*****")
		if args.load_model_name is not None:
			# Set tokenizer
			tokenizer = DistilBertTokenizer.from_pretrained(args.load_model_name, do_lower_case = True)
			# Set model
			model = DistilBertForSequenceClassification.from_pretrained(args.load_model_name, num_labels = 2)
			model.to(device)
			logger.info("Model loaded: %s" % args.load_model_name)
		else:
			logger.error('No model loaded!')

		test_inputs = read_test_data(args.data_dir + 'test_text.tsv', tokenizer, 32)
		print(len(test_inputs))
		test_dataloader = DataLoader(test_inputs, sampler = SequentialSampler(test_inputs),
		                             batch_size = args.test_batch_size)

		logger.info("***** Predicting *****")
		# Tracking variables
		predictions = []
		for batch in tqdm(test_dataloader, desc = "Testing"):
			# Add batch to GPU
			batch = tuple(t.to(device) for t in batch)

			# Unpack the inputs from our dataloader
			b_input_ids, b_input_mask = batch
			b_input_ids = b_input_ids.to(device)
			b_input_mask = b_input_mask.to(device)

			# Telling the model not to compute or store gradients, saving memory and
			# speeding up prediction
			with torch.no_grad():
				# Forward pass, calculate logit predictions
				outputs = model(b_input_ids, 
				                attention_mask = b_input_mask)

			logits = outputs[0]

			# Move logits and labels to CPU
			logits = logits.detach().cpu().numpy()
			# Store predictions and true labels
			predictions.append(logits)

		# Combine the results across all batches.
		flat_predictions = np.concatenate(predictions, axis = 0)
		print(len(flat_predictions))
		# For each sample, pick the label (0 or 1) with the higher score.
		flat_predictions = list(np.argmax(flat_predictions, axis = 1).flatten())

		pred_labels = ['machine' if p == 1 else 'human' for p in flat_predictions]
		print(len(pred_labels))

		with open(os.path.join(args.output_dir, 'test_predictions.csv'), 'w') as writer:
			writer.write('Id,Category\n')
			i = 0
			for p in pred_labels:
				writer.write('%d,%s\n' % (i, p))
				i += 1
			writer.close()

	if args.do_test_generation:

		# load model
		logger.info("***** Load test model*****")
		if args.load_model_name is not None:
			# Set tokenizer
			tokenizer = DistilBertTokenizer.from_pretrained(args.load_model_name, do_lower_case = True)
			# Set model
			model = DistilBertForSequenceClassification.from_pretrained(args.load_model_name, num_labels = 2)
			model.to(device)
		else:
			logger.error('No model loaded!')

		learning_rate = float(args.load_model_name.split('-')[2][2:] + '-' + args.load_model_name.split('-')[3])
		length = int(args.load_model_name.split('-')[1][3:])
		logger.info(" Sequence length = %d" % length)
		logger.info(" Learning rate = %f" % learning_rate)

		logger.info("***** Load test data*****")
		# Read and save training input_ids
		text = []
		with open(os.path.join(
				'/home/liyixi/hw5/output/generation/%s' % args.generate_text_name)) as input:
			for _ in range(20):
				sent = input.readline().lstrip().rstrip()
				if len(sent) != 0:
					text.append(sent)
		input.close()
		labels = [1] * len(text)
		print(len(text))
		preds = []
		for sent in text:
			input_ids = torch.tensor(tokenizer.encode(sent, add_special_tokens=True)).unsqueeze(0)
			labels = torch.tensor([1]).unsqueeze(0)
			input_ids = input_ids.to(device)
			labels = labels.to(device)
			output = model(input_ids=input_ids,labels=labels)
			logits = output[1]
			if len(preds) == 0:
				preds.append(logits.detach().cpu().numpy())
			else:
				preds[0] = np.append(
					preds[0], logits.detach().cpu().numpy(), axis = 0)

		acc = flat_accuracy(preds, labels)

		logger.info("***** Test results *****")
		# Report the final accuracy for this test run.
		logger.info("average accuracy: %f" % acc)
		# Measure how long the eval run took.

		preds = preds[0]
		preds = preds.argmax(1)

		output_test_pred_file = os.path.join(args.output_dir, '%s-test_generation_pred.csv' % args.task_name)
		with open(output_test_pred_file, "w") as writer:
			writer.write("Id,Prediction,Generate_text\n")
			for i in range(len(preds)):
				writer.write("%d,%d,%s\n" % (i, preds[i], text[i]))
