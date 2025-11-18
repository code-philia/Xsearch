

# XSearch

## Install Dependencies

```bash
pip install torch
pip install transformers
pip install tree_sitter
pip install hanlp_restful
pip install scikit-learn
export WANDB_API_KEY=<your_wandb_api_key> # refer to (https://docs.wandb.ai/quickstart/)
pip install wandb
wandb login
```

## Data Preprocessing

Our dataset is built upon the [**CodeSearchNet** benchmark](https://github.com/github/CodeSearchNet). 

Input `tune_out_dep.jsonl` should be in **json line format** with each line having the following entries:

```
"repo": the code respository ID
"path": the path to the code file
"func_name": the function name
"url": full url to the code function
"language": program language, e.g. python, java, php
"sha":
"partition": "train" or "valid" or "test"
"original_string": original code
"code": same as "original_string"
"code_tokens": tokenized code with docstring removed
"docstring": original docstring (comment)
"docstring_tokens": tokenized docstring
"docstring_dep": docstring dependency tree graph (from [Hanlp](https://github.com/hankcs/HanLP) Semantic Dependency Parsing API)
"response": a Dict in str format 
           `COMMENT_CONCEPTS` represents the comment tokens corresponding to important concepts
           `ALIGNMENT_MAP` represents the code tokens corresponding to important concepts
```


## Training

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
--data_flow_length 64 \
--nl_length 128 \
--train_batch_size 32 \
--eval_batch_size 32 \
--learning_rate 2e-5 \
--seed 123456 \
2>&1| tee train.log
```

## Evaluation

XSearch employs a comprehensive evaluation framework that assesses both highlight accuracy and alignment accuracy. The evaluation process involves two main components: **Explainable Comment-to-Code Retrieval** and **Alignment and Highlight Accuracy**.

### Explainable Comment-to-Code Retrieval

The retrieval process takes a query comment and a reference codebase to retrieve the most semantically relevant code snippet.

#### Processing Query Comments

1. **Salient Token Identification**: A trained highlight predictor is applied to each token (`n_i`) in the query comment, yielding a probability score (`p_i`). Tokens with `p_i` exceeding a predefined threshold (`δ_highlight`) are marked as salient.

2. **Clustering Salient Tokens**: The embeddings of these salient tokens are extracted and clustered using agglomerative hierarchical clustering with cosine similarity as the linkage criterion.

3. **Cluster Merging**: Clusters are iteratively merged if the cosine similarity between their centroids exceeds another threshold (`δ_cluster`).

4. **Comment Concept Representation**: Each resulting cluster `C` is represented by its centroid:
   $$h_C = \frac{1}{|C|} \sum_{n_i \in C} f_θ(n_i)$$
   where `f_θ(n_i)` is the embedding of token `n_i`.

#### Processing Code Snippets

1. **Salient Token Identification in Code**: A similar procedure is applied to each code snippet to predict salient tokens.

2. **Forming Contiguous Spans**: Adjacent salient tokens on the same source line are merged into contiguous spans (`C_l`).

3. **Line-Level Embedding**: For each span `C_l` on line `l`, a line-level embedding is computed:
   $$e_l = \frac{1}{|C_l|} \sum_{c_j \in C_l} f_θ(c_j)$$

4. **Clustering Code Concepts**: These line-level embeddings are clustered using the same agglomerative procedure, producing clusters `D` summarized by their centroids:
   $$h_D = \frac{1}{|D|} \sum_{e_l \in D} e_l$$

### Alignment and Highlight Accuracy

#### Evaluation Metrics

**Highlight Accuracy**

Highlight accuracy measures the precision and recall of important tokens identified by the model:

- **Token Selection**: A threshold is used to determine if a token's highlight score is considered "important"
- **Ground Truth**: Annotated labels serve as the ground truth for important tokens
- **Metrics**: The final evaluation metric is the average precision and recall across all test samples

**Highlight Precision and Recall**:
- **Highlight Precision (Comment)**: 
  $$\text{Precision}_{NL} = \frac{|\text{predicted comment tokens} \cap \text{ground truth comment tokens}|}{|\text{predicted comment tokens}|}$$

- **Highlight Recall (Comment)**: 
  $$\text{Recall}_{NL} = \frac{|\text{predicted comment tokens} \cap \text{ground truth comment tokens}|}{|\text{ground truth comment tokens}|}$$

- **Highlight Precision (Code)**: 
  $$\text{Precision}_{Code} = \frac{|\text{predicted code tokens} \cap \text{ground truth code tokens}|}{|\text{predicted code tokens}|}$$

- **Highlight Recall (Code)**: 
  $$\text{Recall}_{Code} = \frac{|\text{predicted code tokens} \cap \text{ground truth code tokens}|}{|\text{ground truth code tokens}|}$$

**Jaccard Similarity**:
- **Highlight Jaccard (Comment)**: 
  $$\text{Jaccard}_{NL} = \frac{|\text{predicted comment} \cap \text{ground truth comment}|}{|\text{predicted comment} \cup \text{ground truth comment}|}$$

- **Highlight Jaccard (Code)**: 
  $$\text{Jaccard}_{Code} = \frac{|\text{predicted code} \cap \text{ground truth code}|}{|\text{predicted code} \cup \text{ground truth code}|}$$

**Alignment Accuracy**

Alignment accuracy evaluates how well the model aligns comment concepts with their corresponding code statements:

- **Similarity Computation**: The similarity between each comment concept token and all code statement representations is computed
- **Threshold-based Selection**: If a code statement's similarity exceeds a predefined threshold, it is considered an aligned candidate
- **Ground Truth Comparison**: The average alignment precision and recall are measured by comparing predicted aligned code statements with annotated ground truth

**Alignment Precision and Recall**:
- **Alignment Precision**: 
  $$\text{Alignment Precision} = \max_{comment_i, concept_c}{\frac{|\text{predicted topk code} \cap \text{ground truth code}|}{|\text{predicted topk code}|}}$$

- **Alignment Recall**: 
  $$\text{Alignment Recall} = \max_{comment_i, concept_c}{\frac{|\text{predicted topk code} \cap \text{ground truth code}|}{|\text{ground truth code}|}}$$

**Alignment Recall@k**: 
This metric evaluates whether the top-k most similar code statements retrieved for each comment concept token include any of the annotated ground-truth alignments. This reflects the model's top-k alignment effectiveness in capturing correct semantic alignments between comments and code.

### Evaluation Commands

#### Highlight and Alignment Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python eval_loss.py \
--output_dir ./checkpoints/python \
--config_name microsoft/graphcodebert-base \
--model_name_or_path microsoft/graphcodebert-base \
--tokenizer_name microsoft/graphcodebert-base \
--lang python \
--test_data_file ./dataset/python/test.jsonl \
--roles_file ./dataset/python/roles.npy \
--code_length 256 --data_flow_length 64 --nl_length 128
```

#### Retrieval Evaluation
```bash
CUDA_VISIBLE_DEVICES=0 python eval_retrieval.py \
--output_dir ./checkpoints/python \
--config_name microsoft/graphcodebert-base \
--model_name_or_path microsoft/graphcodebert-base \
--tokenizer_name microsoft/graphcodebert-base \
--lang python \
--test_data_file ./dataset/python/test.jsonl \
--codebase_data_file ./dataset/python/codebase.jsonl \
--roles_file ./dataset/python/roles.npy \
--code_length 256 --data_flow_length 64 --nl_length 128
```

Model output during inference:
```python
class OutputFeatures:
    code_ids: Optional[Tensor]  # input code tokens IDs [B, L_code,]
    code_hidden: Optional[Tensor]  # predicted last hidden state [B, L_code, H]
    code_scores: Optional[Tensor]  # predicted highlight scores [B, L_code, ]
    code_indices: Optional[List[List[int]]] # predicted highlight positions, list of size B
    nl_ids: Optional[Tensor]  # input comment tokens IDs [B, L_nl,]
    nl_hidden: Optional[Tensor]  # predicted last hidden state [B, L_nl, H]
    nl_scores: Optional[Tensor]  # predicted highlight scores [B, L_nl, ]
    nl_indices: Optional[List[List[int]]] # predicted highlight positions, list of size B
    sim_mat: Optional[Tensor] # similarity matrix [B, L_nl, L_code]
```
