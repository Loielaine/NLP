# Folder Structure
hw5

|____data

| |____classification

| |____generation

|____output

| |____classification

| |____fine-tuning

| |____generation

|____script

| |____classification

| |____generation

|____readme.txt

 
# Task 1 Generation

model: script/generation/model_openai_gpt.py (openai_gpt)

(1) train and eval
    - command file
      script/generation/run_openai_gpt_train.sbat

    - parameters for validation
      length='32 64 96'
      learning_rate=5e-05

    - data
      data/generation/train.text dev.txt

    - log
      output/generation/log_openai_gpt_valid_20200424-233257.log

    - saved results
      output/generation/openai_gpt_valid_result.log

    - saved models
      output/generation/openai_gpt_valid-len32-lr5e-05-epoch4
      output/generation/openai_gpt_valid-len64-lr5e-05-epoch4
      output/generation/openai_gpt_valid-len96-lr5e-05-epoch4

(2) test and generate
    - command file
      script/generation/run_openai_gpt_test_generate.sbat

    - data
      data/generation/test.txt

    - log & saved results
      output/generation/log_openai_gpt_test_20200425-013713.log
      output/generation/log_openai_gpt_test_20200425-014411.log

    - generate texts
      output/generation/openai_gpt_valid-len64-lr5e-05-epoch4-generate_text.text
      output/generation/openai_gpt_valid-len96-lr5e-05-epoch4-generate_text.text


# Task 2.1 Pre-training

model: script/classification/model_pretraining_auto.py (distilbert)

(1) train and eval
    - command file
      script/classification/run_pretraining_distillbert.sbat

    - data
      data/generation/train.text

    - log
      output/fine-tuning/log_pretraining_distilbert_20200426-164712.log

    - result
       output/fine-tuning/eval_results.txt

    - saved model
      output/fine-tuning/


# Task 2.2 Classification

model: script/classification/model_classification.py (distilbert)

(1）train and eval
    - command file
      without pre-training: script/classification/run_classification.sbat
      with pre-training: script/classification/run_pretraining_classification.sbat

    - data
      data/classification/train.tsv dev.tsv

    - parameters for validation
      length=32
      learning_rate='5e-05'

    - log
      without pre-training: output/classification/log_bert_classification_20200425-182835.log
      with pre-training: output/classification/log_pretraining_classification_distilbert_20200427-015841.log

    - results
      without pre-training: output/classification/bert_classification_result.log
      with pre-training: output/classification/pretraining_classification_distilbert_result.log

    - saved models
      without pre-training: output/classification/bert_classification-len32-lr5e-05-epoch4
      with pre-training: output/classification/pretraining_classification_distilbert-len32-lr5e-05-epoch4


(2) test
    - command file
      script/classification/run_classification_test.sbat

    - data
      data/classification/test.tsv test_text.tsv

    - results
      without pre-training: output/classification/bert_test_predictions.csv
      with pre-training: output/classification/distilbert_pretrain_test_predictions.csv


(3) test generated sentences
    - command file
      script/classification/run_classification_test.sbat

    - results
      output/classification/test_generation_pred1.csv
      output/classification/test_generation_pred2.csv

