lang=python

CUDA_VISIBLE_DEVICES=1,2,3 python run.py \
--output_dir ./checkpoints/python \
--config_name microsoft/graphcodebert-base \
--model_name_or_path microsoft/graphcodebert-base \
--tokenizer_name microsoft/graphcodebert-base \
--lang=$lang \
--do_train \
--train_data_file=./dataset/$lang/train.jsonl \
--eval_data_file=./dataset/$lang/valid.jsonl \
--test_data_file=./dataset/$lang/test.jsonl \
--codebase_file=./dataset/$lang/codebase.jsonl \
--num_train_epochs 10 \
--code_length 256 \
--data_flow_length 64 \
--nl_length 128 \
--train_batch_size 128 \
--eval_batch_size 128 \
--learning_rate 2e-5 \
--seed 123456 \
2>&1| tee train.log