# XSearch

XSearch is an explainable comment-to-code retrieval framework that supports training and evaluation of:
- Highlighting: identifying salient tokens in comments and code
- Alignment: aligning comment concepts with corresponding code statements
- Retrieval: ranking relevant code snippets for a given query comment


---

## Website

> More project details and qualitative examples are available on our anonymous project page:
> https://xsearch052-ux.github.io/xsearch.github.io


---

## Environment

We recommend running XSearch on Linux.

- Hardware
  - GPU recommended (CUDA) for training and retrieval evaluation

- Software
  - Python 3.8+
  - PyTorch + Transformers
  - tree_sitter
  - scikit-learn
  - hanlp_restful (for dependency parsing, if used)
  - (Optional) Weights & Biases for logging

---

## Setup

### 1) Install Dependencies

pip install torch
pip install transformers
pip install tree_sitter
pip install hanlp_restful
pip install scikit-learn

# (optional) W&B logging
export WANDB_API_KEY=<your_wandb_api_key>  # refer to https://docs.wandb.ai/quickstart/
pip install wandb
wandb login

### 2) Prepare Data

> **Dataset & Checkpoints**  
> You can download our annotated dataset and pretrained model checkpoints from:
> https://drive.google.com/drive/folders/12Qob4dyzYts8SLKVNDWtERCY4cN4soKI?usp=sharing  

Make sure you have prepared the dataset files referenced in the commands below, e.g.:

- ./dataset/python/train.jsonl
- ./dataset/python/valid.jsonl
- ./dataset/python/test.jsonl
- ./dataset/python/codebase.jsonl
- ./dataset/python/roles.npy

---

## Usage

### Training
```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --output_dir ./checkpoints/python \
  --config_name microsoft/graphcodebert-base \
  --model_name_or_path microsoft/graphcodebert-base \
  --tokenizer_name microsoft/graphcodebert-base \
  --lang=python \
  --do_train \
  --train_data_file=./dataset/$lang/train.jsonl \
  --eval_data_file=./dataset/$lang/valid.jsonl \
  --test_data_file=./dataset/$lang/test.jsonl \
  --codebase_file=./dataset/$lang/codebase.jsonl \
  --num_train_epochs 10 \
  --code_length 256 \
  --data_flow_length 0 \
  --nl_length 128 \
  --train_batch_size 32 \
  --eval_batch_size 32 \
  --learning_rate 2e-5 \
  --seed 123456 \
  2>&1 | tee train.log
```

### Evaluation

XSearch provides a comprehensive evaluation framework covering two components:
- Explainable Comment-to-Code Retrieval
- Alignment and Highlight Accuracy


1) Retrieval Evaluation
> **Using the provided data/checkpoints.**  
> If you download the `data/` and `checkpoints/` folders from our Google Drive and place them under the XSearch project root (i.e., `./data` and `./checkpoints`), you can run retrieval evaluation with the following command:
```bash
python eval_retrieval.py \
  --packed_cache_file data/packed_codebase_cache.pt \
  --test_data_file data/test_py.jsonl \
  --codebase_data_file data/python_codebase.jsonl \
  --roles_file data/role_tensor_python_codebase.npy \
  --xsearch_ckpt checkpoints/subject_model_python.pth \
  --device cuda:1 \
  --eval_batch_size 32 \
  --cache_batch_size 256 \
  --code_length 256 \
  --max_code_clusters 64 \
  --code_prob_threshold 0.4 \
  --highlight_threshold 0.4 \
  --cluster_threshold 0.8 \
  --pool_workers 16 \
  --no_per_code_cache
```

2) Highlight + Alignment Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python eval_loss.py \
  --output_dir ./checkpoints/python \
  --config_name microsoft/graphcodebert-base \
  --model_name_or_path microsoft/graphcodebert-base \
  --tokenizer_name microsoft/graphcodebert-base \
  --lang python \
  --test_data_file ./dataset/python/test.jsonl \
  --roles_file ./dataset/python/roles.npy \
  --code_length 256 --data_flow_length 0 --nl_length 128
```
