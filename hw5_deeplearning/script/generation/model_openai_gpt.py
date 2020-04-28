import argparse
import logging
import os
import pickle
import time
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange



def read_data(filename, max_seq_length,args):
	text = []
	with open(filename, 'r') as f:
		line = f.readline()
		while True:
			line = f.readline()
			text.append(line)
			if not line:
				break

	input_ids = []
	# For every sentence in data text
	for sent in text:
		encoded_dict = tokenizer.encode_plus(
			sent,
			add_special_token = True,
			max_length = max_seq_length,
			pad_to_max_length = True,
			return_attention_mask = False,
			return_tensors = 'pt'
		)

		# Add the encoded sentence to the list.
		input_ids.append(torch.tensor(encoded_dict['input_ids'], dtype = torch.long))
		
	# Convert input_ids and label into tensors
	input_ids = torch.cat(input_ids, dim = 0)
	input_ids, labels = mask_tokens(input_ids,tokenizer,args)
	
	# Convert dataset into TensorDataset format
	dataset = TensorDataset(input_ids, labels)
	return dataset


def mask_tokens(inputs, tokenizer, args):
	""" Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """


	if tokenizer.mask_token is None:
		raise ValueError(
			"This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
		)

	labels = inputs.clone()
	# We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
	probability_matrix = torch.full(labels.shape, args.mlm_probability)
	special_tokens_mask = [
		tokenizer.get_special_tokens_mask(val, already_has_special_tokens = True) for val in labels.tolist()
	]
	probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype = torch.bool), value = 0.0)
	if tokenizer._pad_token is not None:
		padding_mask = labels.eq(tokenizer.pad_token_id)
		probability_matrix.masked_fill_(padding_mask, value = 0.0)
	masked_indices = torch.bernoulli(probability_matrix).bool()
	labels[~masked_indices] = -100  # We only compute loss on masked tokens

	# 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
	indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
	inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

	# 10% of the time, we replace masked input tokens with random word
	indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
	random_words = torch.randint(len(tokenizer), labels.shape, dtype = torch.long)
	inputs[indices_random] = random_words[indices_random]

	# The rest of the time (10% of the time) we keep the masked input tokens unchanged
	return inputs , labels

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# Required parameters
	parser.add_argument("--data_dir",
	                    default = '/home/liyixi/hw5/data/generation/',
	                    type = str)
	parser.add_argument("--task_name",
	                    default = None,
	                    type = str,
	                    required = True,
	                    help = "The name of the task to train.")
	parser.add_argument("--output_dir",
	                    default = '/home/liyixi/hw5/output/generation/',
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
	parser.add_argument("--do_generate",
	                    action = 'store_true',
	                    help = "Whether to run generate on the test set.")
	parser.add_argument("--train_batch_size",
	                    default = 32,
	                    type = int,
	                    help = "Total batch size for training.")
	parser.add_argument("--eval_batch_size",
	                    default = 8,
	                    type = int,
	                    help = "Total batch size for eval.")
	parser.add_argument("--test_batch_size",
	                    default = 1,
	                    type = int,
	                    help = "Total batch size for test.")
	parser.add_argument("--learning_rate", nargs = '+',
	                    default = [],
	                    type = str,
	                    help = "The initial learning rate for optimizer.")
	parser.add_argument("--num_train_epochs",
	                    default = 5.0,
	                    type = float,
	                    help = "Total number of training epochs to perform.")
	parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
	parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
	parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
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
	parser.add_argument("--length", nargs = '+', type = str, default = [])
	parser.add_argument("--generate_seq_length", type = int, default = 64)
	parser.add_argument("--stop_token", type = str, default = None, help = "Token at which text generation is stopped")
	parser.add_argument("--seed", type = int, default = 9001, help = "random seed for initialization")
	parser.add_argument(
		"--temperature",
		type = float,
		default = 0.9,
		help = "temperature of 1.0 has no effect, lower tend toward greedy sampling",
	)
	parser.add_argument(
		"--repetition_penalty", type = float, default = 1.2,
		help = "primarily useful for CTRL model; in that case, use 1.2"
	)
	parser.add_argument("--k", type = int, default = 5)
	parser.add_argument("--p", type = float, default = 0.9)
	parser.add_argument(
		"--load_model_name",
		help = "Load the specified " + "saved model for testing",
		type = str,
		default = None)
	parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
	parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
	args = parser.parse_args()
	np.random.seed(args.seed)

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
		# Set tokenizer
		tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

		# Parameters for cross-validation
		length_list = list(map(int, args.length[0].split(' ')))
		lr_list = list(map(float, args.learning_rate[0].split(' ')))
		print(length_list)
		print(lr_list)

		for (length, learning_rate) in [(i, j) for i in length_list for j in lr_list]:
			logger.info("***** Validation parameters*****")
			logger.info(" Sequence length = %d" % length)
			logger.info(" Learning rate = %f" % learning_rate)
			logger.info("***** Load training data*****")
			# Read and save training input_ids
			cached_input_file = os.path.join(args.data_dir, 'train_{0}_{1}'.format(
				str(args.task_name), str(length)))
			try:
				with open(cached_input_file, "rb") as reader:
					train_inputs = pickle.load(reader)
					reader.close()
			except:
				train_inputs = read_data(args.data_dir + 'train.txt', length)
				if args.local_rank == -1:
					logger.info("  Saving train features into cached file %s", cached_input_file)
					with open(cached_input_file, "wb") as writer:
						pickle.dump(train_inputs, writer)
						writer.close()

			train_dataloader = DataLoader(train_inputs, sampler = RandomSampler(train_inputs),
			                              batch_size = args.train_batch_size)

			logger.info("***** Load eval data*****")
			# Read and save training input_ids
			cached_input_file = os.path.join(args.data_dir, 'dev_{0}_{1}'.format(
				str(args.task_name), str(length)))
			try:
				with open(cached_input_file, "rb") as reader:
					eval_inputs = pickle.load(reader)
					reader.close()
			except:
				eval_inputs = read_data(args.data_dir + 'dev.txt', length)
				if args.local_rank == -1:
					logger.info("  Saving eval features into cached file %s", cached_input_file)
					with open(cached_input_file, "wb") as writer:
						pickle.dump(eval_inputs, writer)
						writer.close()

			eval_dataloader = DataLoader(eval_inputs, sampler = SequentialSampler(eval_inputs),
			                             batch_size = args.eval_batch_size)

			# Set model
			model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
			model.to(device)

			# Train model
			logger.info("***** Run training and evaluating *****")
			logger.info("  Num of train examples = %d", len(train_dataloader))
			logger.info("  Train batch size = %d", args.train_batch_size)
			logger.info("  Num of eval examples = %d", len(eval_dataloader))
			logger.info("  Eval batch size = %d", args.eval_batch_size)
			model.train()

			num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs // args.train_batch_size

			# Prepare optimizer and schedule (linear warmup and decay)
			no_decay = ["bias", "LayerNorm.weight"]
			optimizer_grouped_parameters = [
		        {
		            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
		            "weight_decay": args.weight_decay,
		        },
		        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
		    ]
			optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=args.adam_epsilon)
			scheduler = get_linear_schedule_with_warmup(
		        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_train_optimization_steps
		    )

			tr_loss = 0.0
			global_step = 0
			loss_list = defaultdict(list)
			train_perplexity_list = defaultdict(list)
			eval_perplexity_list = defaultdict(list)
			for epoch in trange(int(args.num_train_epochs), desc = "Epoch", disable = args.local_rank not in [-1, 0]):
				logger.info("epoch: %d", epoch)
				for step, batch in enumerate(
						tqdm(train_dataloader, desc = "Iteration", disable = args.local_rank not in [-1, 0])):
					batch = tuple(t.to(device) for t in batch)
					input_ids,  lm_labels = batch

					input_ids = input_ids.to(device)
					lm_labels = lm_labels.to(device)

					# define a new function to compute loss values for both output_modes
					outputs = model(input_ids = input_ids,
					                labels = lm_labels)

					loss = outputs[0]
					loss.backward()
					tr_loss += loss.item()
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
					logger.info("epoch %d step %d loss: %f" % (epoch, step, loss.item()))
					optimizer.step()
					scheduler.step()
					optimizer.zero_grad()
					global_step += 1

				# save model for each epochs
				# only save the last model:
				if epoch == args.num_train_epochs - 1:
					model_name = '%s-len%d-lr%s-epoch%d' % (args.task_name, length, str(learning_rate), epoch)
					output_model_file = os.path.join(args.output_dir, model_name)
					if not os.path.exists(output_model_file):
						os.makedirs(output_model_file)

					model_to_save = model.module if hasattr(model,
					                                        'module') else model  # Take care of distributed/parallel training
					model_to_save.save_pretrained(output_model_file)
					tokenizer.save_pretrained(output_model_file)

				loss_list[epoch].append(tr_loss/global_step)
				train_perplexity = torch.exp(torch.tensor(tr_loss/global_step))
				logger.info("epoch %d train perplexity: %f" % (epoch, train_perplexity.item()))
				train_perplexity_list[epoch].append(train_perplexity.item())

				batch_perplexity_list = []
				for input_ids,  lm_labels in tqdm(eval_dataloader, desc = "Evaluating"):
					input_ids = input_ids.to(device)
					lm_labels = lm_labels.to(device)
					with torch.no_grad():
						outputs = model(input_ids = input_ids,
						                labels = lm_labels)
						loss = outputs[0]
						batch_perplexity = torch.exp(torch.tensor(loss))

					batch_perplexity_list.append(batch_perplexity.item())

				eval_perplexity = sum(batch_perplexity_list) / len(batch_perplexity_list)
				logger.info("epoch %d eval perplexity: %f" % (epoch, eval_perplexity))
				eval_perplexity_list[epoch].append(eval_perplexity)

			logger.info("***** Training and evaluating results *****")
			logger.info("loss on training data: ")
			logger.info(loss_list)
			logger.info("perplexity on training data: ")
			logger.info(train_perplexity_list)
			logger.info("perplexity on eval data: ")
			logger.info(eval_perplexity_list)

			# Save results to file
			output_result_file = os.path.join(args.output_dir, '%s_result.log' % args.task_name)
			with open(output_result_file, "a") as writer:
				writer.write("Sequence length: %d\n" % length)
				writer.write("Learning rate: %s\n" % str(learning_rate))

				for i in range(int(args.num_train_epochs)):
					writer.write('Epoch: %d\n' % i)
					writer.write("training loss: %1.3f, training perplexity: %1.3f, eval perplexity: %1.3f" \
					             % (loss_list[i][-1], train_perplexity_list[i][-1], eval_perplexity_list[i][-1]))
					writer.write('\n')
				writer.write('\n')

	if args.do_test:

		# load model and tokenizer
		logger.info("***** Load test model*****")
		if args.load_model_name is not None:
			load_model_file = os.path.join(args.output_dir, args.load_model_name)
			model = OpenAIGPTLMHeadModel.from_pretrained(load_model_file)
			model.to(device)
			tokenizer = OpenAIGPTTokenizer.from_pretrained(load_model_file)
		else:
			logger.error('No model loaded!')

		learning_rate = float(args.load_model_name.split('-')[2][2:] + '-' + args.load_model_name.split('-')[3])
		length = int(args.load_model_name.split('-')[1][3:])
		# learning_rate = 5e-05
		# length = 32
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
			test_inputs = read_data(args.data_dir + 'test.txt', length)

			logger.info("***** Build test features *****")

			if args.local_rank == -1:
				logger.info("  Saving eval features into cached file %s", cached_input_file)
				with open(cached_input_file, "wb") as writer:
					pickle.dump(test_inputs, writer)
					writer.close()

		test_dataloader = DataLoader(test_inputs, sampler = SequentialSampler(test_inputs),
		                             batch_size = args.test_batch_size)
		test_perplexity_list = []
		for input_ids,  lm_labels in tqdm(test_dataloader, desc = "Testing"):
			input_ids = input_ids.to(device)
			lm_labels = lm_labels.to(device)
			with torch.no_grad():
				outputs = model(input_ids = input_ids,
				                labels = lm_labels)
				loss = outputs[0]
				perplexity = torch.exp(torch.tensor(loss))
				# logger.info(perplexity)
				test_perplexity_list.append(perplexity.item())

		test_perplexity = sum(test_perplexity_list) / len(test_perplexity_list)
		logger.info("perplexity on test data: %1.3f" % test_perplexity)

	if args.do_generate:
		# load model and tokenizer
		logger.info("***** Load generate model*****")
		if args.load_model_name is not None:
			load_model_file = os.path.join(args.output_dir, args.load_model_name)
			model = OpenAIGPTLMHeadModel.from_pretrained(load_model_file)
			# model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
			model.to(device)
			# tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
			tokenizer = OpenAIGPTTokenizer.from_pretrained(load_model_file)
		else:
			logger.error('No model loaded!')
			model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
			tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

		logger.info("***** Load prompts*****")
		prompts = ['My', 'The', 'One', 'When', 'If', 'Our', 'First', 'Natural', 'We', 'Because']

		generate_texts = []
		for p in prompts:
			input_ids = tokenizer.encode(p, add_special_tokens = False, return_tensors = "pt")
			input_ids = input_ids.to(device)
			# Generate sequentially
			output_sequences = model.generate(
				input_ids = input_ids,
				max_length = args.generate_seq_length + len(p),
				temperature = args.temperature,
				top_k = args.k,
				top_p = args.p,
				repetition_penalty = args.repetition_penalty,
				do_sample = True,
				num_return_sequences = 1)

			# Remove the batch dimension when returning multiple sequences
			if len(output_sequences.shape) > 2:
				output_sequences.squeeze_()

			generated_sequences = []
			for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
				print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
				generated_sequence = generated_sequence.tolist()

			# Decode text
			text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces = True)

			# Remove all text after the stop token
			text = text[: text.find(args.stop_token) if args.stop_token else None]

			# Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
			total_sequence = (
					p + text[len(tokenizer.decode(input_ids[0], clean_up_tokenization_spaces = True)):]
			)
			logger.info(total_sequence)
			generate_texts.append(total_sequence)

		with open(os.path.join(args.output_dir, '%s-generate_text.text' % args.load_model_name), 'w') as writer:
			for t in generate_texts:
				writer.write(t + "\n")
				writer.write("\n")
			writer.close()
