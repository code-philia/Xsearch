import os
import torch
import argparse
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (RobertaModel, RobertaTokenizer, RobertaTokenizerFast)
from dataloader import TextDataset, textdataset_collate_fn
from model_ablation_typeemb import Model
import multiprocessing
from typing import Tuple, List, Set
import numpy as np
from tqdm import tqdm
import json
import itertools
import re
from io import StringIO
import tokenize

def get_latest_epoch(folder: str) -> str:
    # List all entries named "Epoch_<num>"
    epochs = [
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d)) and d.startswith("Epoch_")
    ]
    # Pick the one with the highest trailing number
    latest = max(epochs, key=lambda x: int(x.split("_", 1)[1]))
    return latest


def extract_representations(
    test_dataloader: DataLoader,
    codebase_dataloader: DataLoader,
    model: Model,
    args: argparse.Namespace,
    roles: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract representations from both test and codebase datasets.
    Returns:
        Tuple containing:
        - test_nl_repr: Test dataset natural language representations
        - test_code_repr: Test dataset code representations
        - codebase_nl_repr: Codebase dataset natural language representations
        - codebase_code_repr: Codebase dataset code representations
    """
    model.eval()
    
    # Initialize lists to store representations
    test_nl_reprs = []
    test_code_reprs = []
    codebase_nl_reprs = []
    codebase_code_reprs = []
    
    # Process test dataset
    for step, batch in enumerate(test_dataloader):
        code_inputs = batch[0].to(args.device)
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        nl_inputs = batch[3].to(args.device)
        ori2cur_pos = batch[4].to(args.device)
        
        batch_start_idx = step * code_inputs.size(0)
        batch_end_idx = min((step + 1) * code_inputs.size(0), len(roles))
        batch_roles = roles[batch_start_idx:batch_end_idx].to(args.device)
        
        with torch.inference_mode():
            outputs = model(code_inputs=code_inputs, attn_mask=attn_mask, 
                          position_idx=position_idx, nl_inputs=nl_inputs,
                          role_indices=batch_roles)
            
            test_nl_reprs.append(outputs.nl_hidden.detach().cpu())
            test_code_reprs.append(outputs.code_hidden.detach().cpu())
    
    # Process codebase dataset
    for step, batch in enumerate(codebase_dataloader):
        code_inputs = batch[0].to(args.device)
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        nl_inputs = batch[3].to(args.device)
        ori2cur_pos = batch[4].to(args.device)
        
        batch_start_idx = step * code_inputs.size(0)
        batch_end_idx = min((step + 1) * code_inputs.size(0), len(roles))
        batch_roles = roles[batch_start_idx:batch_end_idx].to(args.device)
        
        with torch.inference_mode():
            outputs = model(code_inputs=code_inputs, attn_mask=attn_mask, 
                          position_idx=position_idx, nl_inputs=nl_inputs,
                          role_indices=batch_roles)
            
            codebase_nl_reprs.append(outputs.nl_hidden.detach().cpu())
            codebase_code_reprs.append(outputs.code_hidden.detach().cpu())
    
    # Concatenate all representations
    test_nl_repr = torch.cat(test_nl_reprs, dim=0)
    test_code_repr = torch.cat(test_code_reprs, dim=0)
    codebase_nl_repr = torch.cat(codebase_nl_reprs, dim=0)
    codebase_code_repr = torch.cat(codebase_code_reprs, dim=0)
    
    return test_nl_repr, test_code_repr, codebase_nl_repr, codebase_code_repr


from sklearn.cluster import AgglomerativeClustering

def extract_and_cluster_concepts(hidden_states, probs, is_code=True, cluster_threshold=0.7, device='cuda'):
    """
    Cluster the input hidden_states and probs (regardless of code/comment), and return the clustering results and weights.
    Args:
        hidden_states: Hidden states of shape (seq_len, hidden_size)
        probs: Importance scores of shape (seq_len,)
        cluster_threshold: Threshold for merging clusters
        device: Device to perform computations on
    Returns:
        clusters: List of tuples (cluster_vectors, centroid, original_indices) for each cluster
        weights: List of concept weights (average highlight score) for each cluster
    """
    threshold = 0.5 if is_code else 0.4
    important_indices = sorted((probs > threshold).nonzero(as_tuple=True)[0].tolist())
    if not important_indices:
        return None, None

    vectors = hidden_states[important_indices].to(device)  # [num_tokens, hidden_size]
    vectors = torch.nn.functional.normalize(vectors, dim=1)

    # Special case: if only one token is important, return it as a single cluster
    if len(important_indices) == 1:
        concept_weight = probs[important_indices[0]].item()
        return ([(vectors, vectors[0], important_indices)], [concept_weight])

    # cosine distance = 1 - cosine similarity
    vectors_np = vectors.cpu().numpy()
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='average',
        distance_threshold=1 - cluster_threshold
    )
    labels = clustering.fit_predict(vectors_np)

    # Compute clusters and centroids
    cluster_results = []
    concept_weights = []
    for label in set(labels):
        cluster_indices = np.where(labels == label)[0]
        cluster_original_indices = [important_indices[idx] for idx in cluster_indices]
        cluster_vecs = vectors[cluster_indices]  # [num_tokens_in_cluster, hidden_size]
        centroid = torch.nn.functional.normalize(cluster_vecs.mean(dim=0), dim=0)  # [hidden_size]
        # concept_weight = probs[cluster_original_indices].mean().item()
        cluster_results.append((cluster_vecs, centroid, cluster_original_indices))
        concept_weights.append(0.0)

    return (cluster_results, concept_weights)

def normalize_concept_weights(weights, device='cuda'):
    """
    Normalize concept weights to sum to 1.
    If all weights are zero, use uniform weights.
    
    Args:
        weights: List of concept weights
        device: Device to perform computations on
        
    Returns:
        torch.Tensor: Normalized weights on the specified device
    """
    weights = torch.tensor(weights, device=device)
    if weights.sum() > 0:
        return weights / weights.sum()
    else:
        return torch.ones_like(weights) / len(weights)

def compute_concept_similarity_centroid2centroid(
    nl_centroids, nl_weights, code_centroids, device='cuda'
):
    """
    nl_centroids: [num_nl_clusters, hidden_size]
    nl_weights: [num_nl_clusters]
    code_centroids: [num_code_clusters, hidden_size]
    """
    if nl_centroids is None or code_centroids is None:
        return 0.0

    # [num_nl_clusters, num_code_clusters]
    sim_matrix = torch.matmul(nl_centroids, code_centroids.T)
    # For each comment cluster centroid, find the maximum similarity among code cluster centroids
    max_sims = sim_matrix.max(dim=1)[0]  # [num_nl_clusters]
    # Weighted average using nl_weights
    weighted_sim = (max_sims * nl_weights).sum().item()
    return weighted_sim

def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # Remove docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    
def extract_code_line_clusters_from_example(
    code_tokens, code_hidden, code_probs, raw_code, tokenizer, lang='go', prob_threshold=0.5
):
    code_wo_comments = remove_comments_and_docstrings(raw_code, lang)
    clusters = []
    # 1. Concatenate code_tokens into a list of strings without spaces
    def clean_token_str(s):
        # Clean Ġ, spaces, and strange ĉ symbols
        return s.replace('Ġ', '').replace(' ', '').replace('ĉ', '')
    code_token_strs = [clean_token_str(t) for t in code_tokens]
    N = len(code_token_strs)
    # DEBUG: Print code_token_strs
    # print("[DEBUG] code_token_strs:", code_token_strs)
    for line_no, line in enumerate(code_wo_comments.splitlines()):
        # Clean strange characters at the beginning of the line (such as ĉ in php)
        line_clean = line.replace('ĉ', '')
        line_tokens = tokenizer.tokenize(line_clean)
        line_tokens = [tok for tok in line_tokens if tok != 'Ġ']
        if not line_tokens:
            continue
        # Concatenate tokens of the line into a string without spaces, and clean ĉ
        line_str = ''.join([clean_token_str(t) for t in line_tokens])
        if not line_str:
            continue
        # DEBUG: Print line and its tokenized/stripped version
        # print(f"[DEBUG] Line {line_no}: '{line}' -> tokens: {line_tokens} -> line_str: '{line_str}'")
        found = False
        ptr = 1
        while ptr < N:
            acc = ''
            end = ptr
            while end < N and len(acc) < len(line_str):
                acc += code_token_strs[end]
                end += 1
            if len(acc) == len(line_str) and acc == line_str:
                indices = list(range(ptr, end))
                highlight_indices = [idx for idx in indices if code_probs[idx] > prob_threshold]
                # DEBUG: Print mapping from line to token indices and highlight indices
                # print(f"[DEBUG]   Matched line {line_no} to token indices {indices}, highlight_indices: {highlight_indices}")
                if highlight_indices:
                    cluster_vecs = code_hidden[highlight_indices]
                    centroid = cluster_vecs.mean(dim=0)
                    clusters.append(([], centroid, highlight_indices))
                found = True
                break
            ptr += 1
    return clusters

def evaluate_concept_search(
    test_dataloader: DataLoader,
    codebase_dataloader: DataLoader,
    model: Model,
    args: argparse.Namespace,
    roles: torch.Tensor,
    tokenizer,
    cluster_threshold: float = 0.7
) -> float:
    """
    Evaluate code search using concept matching with comment clustering.
    """
    model.eval()
    device = args.device

    # Store all code representations
    code_representations = []
    code_urls = []
    code_cluster_lens = []
    all_code_centroids_list = []

    roles = roles.to(device)
    print("Processing codebase...")
    # Process codebase first
    for step, batch in enumerate(tqdm(codebase_dataloader, desc="Processing codebase")):
        code_inputs = batch[0].to(device)
        attn_mask = batch[1].to(device)
        position_idx = batch[2].to(device)

        # Get batch roles
        batch_start_idx = step * code_inputs.size(0)
        batch_end_idx = min((step + 1) * code_inputs.size(0), len(roles))
        batch_roles = roles[batch_start_idx:batch_end_idx]

        with torch.inference_mode():
            outputs = model(code_inputs=code_inputs, attn_mask=attn_mask, 
                          position_idx=position_idx, role_indices=batch_roles)
            code_hidden = outputs.code_hidden  # Keep on GPU
            code_probs = outputs.code_scores  # Keep on GPU

            # Extract and cluster code concepts
            batch_code_clusters = []
            for i in range(code_hidden.size(0)):
                code_example = codebase_dataloader.dataset.examples[step * code_hidden.size(0) + i]
                raw_code = codebase_dataloader.dataset.raw_code[step * code_hidden.size(0) + i]
                code_clusters = extract_code_line_clusters_from_example(
                    code_example.code_tokens, code_hidden[i], code_probs[i], raw_code, tokenizer, prob_threshold=0.5
                )
                batch_code_clusters.append((code_clusters, None))
                # centroids
                if code_clusters is not None and len(code_clusters) > 0:
                    centroids = torch.stack([c[1] for c in code_clusters])
                    all_code_centroids_list.append(centroids)
                    code_cluster_lens.append(centroids.shape[0])
                else:
                    code_cluster_lens.append(0)
            code_representations.extend(batch_code_clusters)

        # Get URLs from the dataset examples
        batch_examples = codebase_dataloader.dataset.examples[step * code_inputs.size(0):
                                                             min((step + 1) * code_inputs.size(0), 
                                                                 len(codebase_dataloader.dataset.examples))]
        code_urls.extend([example.code_url for example in batch_examples])

    if len(all_code_centroids_list) > 0:
        all_code_centroids = torch.cat(all_code_centroids_list, dim=0)  # [total_code_clusters, hidden_size]
    else:
        all_code_centroids = None

    print(f"Processed {len(code_representations)} code samples")
    print("\nProcessing test queries...")

    # Process test queries and compute similarities
    ranks = []

    for step, batch in enumerate(tqdm(test_dataloader, desc="Processing test queries")):
        nl_inputs = batch[3].to(device)
        query_example = test_dataloader.dataset.examples[step * nl_inputs.size(0)]
        query_url = query_example.code_url

        with torch.inference_mode():
            outputs = model(nl_inputs=nl_inputs)
            nl_hidden = outputs.nl_hidden  # Keep on GPU
            nl_probs = outputs.nl_scores  # Keep on GPU

        # Extract and cluster comment concepts
        try:
            nl_result = extract_and_cluster_concepts(
                nl_hidden[0], 
                nl_probs[0], 
                is_code=False,  # This will perform clustering
                cluster_threshold=cluster_threshold,
                device=device
            )
            if nl_result is None:
                print(f"Warning: nl_result is None for query {step}")
                continue

            nl_clusters, nl_weights = nl_result

            # Extract centroids
            try:
                nl_centroids = torch.stack([cluster[1] for cluster in nl_clusters])  # [num_clusters, hidden_size]
            except Exception as e:
                print(f"Error processing nl_clusters at step {step}: {str(e)}")
                print(f"nl_clusters type: {type(nl_clusters)}, nl_weights type: {type(nl_weights)}")
                if nl_clusters is not None:
                    print(f"nl_clusters length: {len(nl_clusters)}")
                continue

            similarities = []
            if all_code_centroids is None or all_code_centroids.shape[0] == 0:
                similarities = [0.0 for _ in code_cluster_lens]
            else:
                sim_matrix = torch.matmul(nl_centroids, all_code_centroids.T)  # [num_nl_clusters, total_code_clusters]
                start = 0
                for length in code_cluster_lens:
                    if length == 0:
                        similarities.append(0.0)
                    else:
                        code_sim = sim_matrix[:, start:start+length].max(dim=1)[0]  # [num_nl_clusters]
                        sim_score = code_sim.mean().item()
                        similarities.append(sim_score)
                        start += length
        except Exception as e:
            print(f"Error in main processing loop at step {step}: {str(e)}")
            print(f"nl_hidden shape: {nl_hidden.shape}, nl_probs shape: {nl_probs.shape}")
            continue

        # Get rank of correct code
        if not similarities:  # If all similarities are None or empty
            continue

        sorted_indices = np.argsort(similarities)[::-1]  # Descending order
        try:
            rank = np.where(np.array(code_urls)[sorted_indices] == query_url)[0][0] + 1
            ranks.append(1.0 / rank)
        except IndexError:
            continue

    rank_path = os.path.join("your_dir", 'your_path.json')
    with open(rank_path, 'w') as file:
        json.dump(ranks, file)

    # Calculate MRR
    mrr = np.mean(ranks) if ranks else 0.0
    print(f"\nProcessed {len(ranks)} test queries")
    print(f"cluster threshold: {cluster_threshold}")
    return mrr

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_data_file", default=None, type=str,
                        help="An optional input codebase data file containing code snippets.")
    parser.add_argument("--roles_file", type=str, required=True,
                        help="Path to the roles data file (.npy)")

    parser.add_argument("--lang", default=None, type=str,
                        help="language.")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size for evaluation.")

    args = parser.parse_args()
    pool = multiprocessing.Pool(16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Load roles data
    print("Loading roles data from", args.roles_file)
    roles_data = np.load(args.roles_file)
    roles = torch.from_numpy(roles_data).long()
    print("Loaded roles data with shape:", roles.shape)

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path)
    model = Model(encoder, num_roles=17)  # Initialize with number of roles

    # Load labeled samples first
    test_dataset = TextDataset(tokenizer, args, args.test_data_file, pool, compute_alignment=False)
    codebase_dataset = TextDataset(tokenizer, args, args.codebase_data_file, pool, compute_alignment=False)
    
    # Create dataloaders for both datasets
    test_sampler = SequentialSampler(test_dataset)
    codebase_sampler = SequentialSampler(codebase_dataset)
    
    test_dataloader = DataLoader(test_dataset,
                                sampler=test_sampler,
                                batch_size=1,
                                num_workers=4,
                                collate_fn=textdataset_collate_fn)
                                
    codebase_dataloader = DataLoader(codebase_dataset,
                                    sampler=codebase_sampler,
                                    batch_size=1,
                                    num_workers=4,
                                    collate_fn=textdataset_collate_fn)

    # Evaluate using concept search method with different parameters
    cluster_threshold = 0.8  # Only need cluster threshold for comments
    model_paths = [
        f"your_path.pth"
        for i in range(1, 11)
    ]
    
    best_mrr = 0.0
    best_params = (0.8,)  # Only cluster threshold
    results = {}  # Store results for each parameter combination
    
    print("Starting parameter search...")
    total_combinations = len(model_paths)
    current_combination = 0
    
    for output_dir in model_paths:
        model.load_state_dict(torch.load(output_dir), strict=False)
        model.to(args.device)
        model.eval()
        current_combination += 1
        print(f"\nTesting combination {current_combination}/{total_combinations}")
        try:
            mrr_score = evaluate_concept_search(
                test_dataloader, 
                codebase_dataloader, 
                model, 
                args, 
                roles,
                tokenizer,
                cluster_threshold=cluster_threshold
            )
            print(f"Cluster Threshold: {cluster_threshold:.1f}, "
                  f"MRR: {mrr_score:.4f}")
            
            # Store results
            param_key = f"cluster{cluster_threshold:.1f}"
            results[param_key] = mrr_score
            
            if mrr_score > best_mrr:
                best_mrr = mrr_score
                best_params = (cluster_threshold,)
                print(f"New best parameters found! MRR: {best_mrr:.4f}")
        except Exception as e:
            print(f"Error with parameters cluster_threshold={cluster_threshold}: {str(e)}")
            continue
    
    print("\nParameter search completed!")
    print(f"Best parameters - Cluster Threshold: {best_params[0]:.1f}")
    print(f"Best MRR Score: {best_mrr:.4f}")

if __name__ == "__main__":
    main()
