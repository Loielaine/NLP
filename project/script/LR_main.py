# from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm, trange
from script.LR_model import LogisticRegression
from script.utils import load_data
from script.features import convert_examples_to_features, convert_features_to_tensor
from pytorch_pretrained_bert.tokenization import BertTokenizer

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# Required parameters
	parser.add_argument("--data_dir",
	                    default = '/Users/Loielaine/Desktop/umich-2020/SI630/project/data/',
	                    type = str)
	parser.add_argument("--task_name",
	                    default = None,
	                    type = str,
	                    required = True,
	                    help = "The name of the task to train.")
	parser.add_argument("--output_dir",
	                    default = '/Users/Loielaine/Desktop/umich-2020/SI630/project/output/',
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

	# args = parser.parse_args()
	args = parser.parse_args(['--task_name=LR_four_sentences', '--do_train', '--do_eval', '--do_test'])
	# args = parser.parse_args(['--task_name=LR_four_sentences', '--do_train'])
	# Set device
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

	# Set parameters
	args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	task_name = args.task_name
	label_list = [0, 1]
	num_labels = len(label_list)
	# ngram = args.ngram
	# keep_prob = 0.5
	# hidden_size = 200
	# embedding_dim = 50
	vocab_size = 128
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = args.do_lower_case)
	output_mode = "classification"
	print_every_iters = args.print_every_iters

	model = LogisticRegression(vocab_size, num_labels)
	optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.to(device)
	print("Initiate model -- Done!")

	global_step = 0
	nb_tr_steps = 0
	tr_loss = 0
	if args.do_train:
		# Load data
		train_examples = load_data(args.data_dir, 'train')
		print("Load train examples -- Done!")
		# print("length of training examples:" , len(train_examples))

		# Test with small sample size
		if False:
			train_examples = train_examples[0:100]
		# print("length of training examples:" , len(train_examples))

		cached_train_features_file = os.path.join(args.data_dir, 'train_{0}_{1}'.format(
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
		num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


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
				print(input_ids.size())
				logits = model(input_ids.float())
				loss_fct = nn.CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
				loss.backward()

				# print(logits.view(-1, num_labels))
				# print(logits.argmax(1))

				tr_loss += loss.item()
				nb_tr_examples += input_ids.size(0)
				nb_tr_steps += 1

				if (step + 1) % args.gradient_accumulation_steps == 0:
					optimizer.step()
					optimizer.zero_grad()
					global_step += 1
					# tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
					# tb_writer.add_scalar('loss', loss.item(), global_step)
					if step % print_every_iters == 0:
						loss_list[epoch].append(loss.item())
						acc = int((logits.argmax(1) == label_ids).sum()) / len(label_ids)
						acc_list[epoch].append(acc)
						logger.info('Epoch: %d [%d], loss: %1.3f, acc: %1.3f' \
						            % (epoch, step, loss.item(), acc))

			# save model for each epochs
			output_model_file = os.path.join(args.output_dir,
			                                 '%s-%s-epoch-%d.mdl' % ('LR', task_name, epoch))
			torch.save(model, output_model_file)

		# save train outputs
		output_train_file = os.path.join(args.output_dir, '%s-%s-train_outputs.txt' % ('LR', task_name))
		with open(output_train_file, "w") as writer:
			for i in range(int(args.num_train_epochs)):
				writer.write('Epoch: %d\n' % i)
				for step in range(len(loss_list[i])):
					writer.write("step%d: loss: %1.3f, acc: %1.3f" \
					             % (step, loss_list[i][step], acc_list[i][step]))
					writer.write('\n')
				writer.write('\n')
		# save config
		model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
	# output_config_file = os.path.join(args.output_dir, '%s-%s-config.bin' % (args.bert_model, task_name))
	# model_to_save.config.to_json_file(output_config_file)
	# output_vocab_file = os.path.join(args.output_dir, '%s-%s-vocab.bin' % ('LR', task_name))
	# tokenizer.save_vocabulary(output_vocab_file)

	if args.do_eval:
		# load model
		if args.do_train:
			load_model_file = os.path.join(args.output_dir,
			                               '%s-%s-epoch-%d.mdl' % (
				                               'LR', task_name, int(args.num_train_epochs) - 1))
			model = torch.load(load_model_file)
		elif args.load_model_file is not None:
			model = torch.load(args.load_model_file)
		else:
			model = LogisticRegression(vocab_size, num_labels)

		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = args.do_lower_case)
		device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		model.to(device)

		# Load data
		val_examples = load_data(args.data_dir, 'eval')
		print("Load val examples -- Done!")

		# Test with small sample size
		if False:
			val_examples = val_examples[0:100]

		cached_val_features_file = os.path.join(args.data_dir, 'val_{0}_{1}'.format(
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
		val_dataloader = DataLoader(val_tensor, sampler = RandomSampler(val_tensor),
		                            batch_size = args.eval_batch_size)
		print("Load data -- Done!")

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

			with torch.no_grad():
				logits = model(input_ids.float())

			loss_fct = nn.CrossEntropyLoss()
			tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

			print(logits.view(-1, num_labels))

			eval_loss += tmp_eval_loss.mean().item()
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

		output_eval_pred_file = os.path.join(args.output_dir, '%s-%s-eval_pred.txt' % ('LR', task_name))
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

		output_eval_file = os.path.join(args.output_dir, '%s-%s-eval_outputs.txt' % ('LR', task_name))
		with open(output_eval_file, "w") as writer:
			logger.info("***** Eval results *****")
			for key in sorted(result.keys()):
				logger.info("  %s = %s", key, str(result[key]))
				writer.write("%s = %s\n" % (key, str(result[key])))

	if args.do_test:
		# load model
		if args.do_train:
			load_model_file = os.path.join(args.output_dir,
			                               '%s-%s-epoch-%d.mdl' % (
				                               'LR', task_name, int(args.num_train_epochs) - 1))
			model = torch.load(load_model_file)
		elif args.load_model_file is not None:
			model = torch.load(args.load_model_file)
		else:
			model = LogisticRegression(vocab_size, num_labels)
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = args.do_lower_case)
		device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
		model.to(device)
		# Load data
		test_examples = load_data(args.data_dir, 'test')
		print("Load test examples -- Done!")

		# Test with small sample size
		if False:
			test_examples = test_examples[0:100]

		cached_test_features_file = os.path.join(args.data_dir, 'test_{0}_{1}'.format(
			str(args.max_seq_length),
			str(task_name)))
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
		test_dataloader = DataLoader(test_tensor, sampler = RandomSampler(test_tensor),
		                             batch_size = args.test_batch_size)
		print("Load data -- Done!")

		model.eval()
		test_loss = 0
		nb_test_steps = 0
		preds = []
		out_label_ids = None
		for input_ids, input_mask, segment_ids, label_ids in tqdm(test_dataloader, desc = "Testing"):
			input_ids = input_ids.to(device)
			input_mask = input_mask.to(device)
			segment_ids = segment_ids.to(device)
			label_ids = label_ids.to(device)

			with torch.no_grad():
				logits = model(input_ids.float())

			loss_fct = nn.CrossEntropyLoss()
			tmp_test_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

			test_loss += tmp_test_loss.mean().item()
			nb_test_steps += 1
			if len(preds) == 0:
				preds.append(logits.detach().cpu().numpy())
				out_label_ids = label_ids.detach().cpu().numpy()
			else:
				preds[0] = np.append(
					preds[0], logits.detach().cpu().numpy(), axis = 0)
				out_label_ids = np.append(
					out_label_ids, label_ids.detach().cpu().numpy(), axis = 0)

		test_loss = test_loss / nb_test_steps
		preds = preds[0]
		preds = preds.argmax(1)

		output_test_pred_file = os.path.join(args.output_dir, '%s-%s-test_pred.txt' % ('LR', task_name))
		with open(output_test_pred_file, "w") as writer:
			logger.info("***** Test predictions *****")
			writer.write("Id,Prediction,Label\n")
			for i in range(len(preds)):
				writer.write("%d,%d,%d\n" % (i, preds[i], out_label_ids[i]))

		acc = (preds == out_label_ids).mean()
		loss = tr_loss / global_step if args.do_train else None

		result = {}
		result['test_loss'] = test_loss
		result['global_step'] = global_step
		result['loss'] = loss
		result['accuracy'] = acc

		output_test_file = os.path.join(args.output_dir, '%s-%s-test_outputs.txt' % ('LR', task_name))
		with open(output_test_file, "w") as writer:
			logger.info("***** Test results *****")
			for key in sorted(result.keys()):
				logger.info("  %s = %s", key, str(result[key]))
				writer.write("%s = %s\n" % (key, str(result[key])))
