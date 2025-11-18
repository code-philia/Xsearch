import os
import torch
import argparse
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (RobertaModel, RobertaTokenizer, RobertaTokenizerFast)
from dataloader import TextDataset, textdataset_collate_fn
from model import Model
import multiprocessing
from typing import Tuple, List, Set
import numpy as np
import json

def get_latest_epoch(folder: str) -> str:
    # list all entries named "Epoch_<num>"
    epochs = [
        d for d in os.listdir(folder)
        if os.path.isdir(os.path.join(folder, d)) and d.startswith("Epoch_")
    ]
    # pick the one with the highest trailing number
    latest = max(epochs, key=lambda x: int(x.split("_", 1)[1]))
    return latest


def jaccard_index(pred_indices: set, true_indices: set) -> Tuple[float, float, float]:
    """Compute Jaccard = |A∩B| / |A∪B| (return 0 if both empty)."""
    if (not pred_indices) and (not true_indices):
        return None, None, None

    inter = pred_indices & true_indices
    union = pred_indices | true_indices
    prec = None if not pred_indices else len(inter) / len(pred_indices) # no prediction
    rec  = None if not true_indices else len(inter) / len(true_indices) # no ground-truth

    return len(inter) / len(union), prec, rec


from tqdm import tqdm

def evaluate_metrics(
    train_dataloader: DataLoader,
    full_dataset: TextDataset,
    model: Model,
    args: argparse.Namespace,
    roles: torch.Tensor,
) -> Tuple[Tuple[List[float], List[float]], Tuple[List[float], List[float], List[float]], Tuple[List[float], List[float], List[float]], List[float], Tuple[int, int], Tuple[int, int]]:

    jacc_nl_all: List[float] = []
    precision_nl_all: List[float] = []
    recall_nl_all: List[float] = []
    jacc_code_all: List[float] = []
    precision_code_all: List[float] = []
    recall_code_all: List[float] = []

    align_precision_all: List[float] = []
    align_recall_all: List[float] = []
    
    # New metric for retrieval specific attention
    retrieval_recall_all: List[float] = []
    retrieval_details_all: List[Tuple[int, float, List[Tuple[int, int]]]] = []  # (index, recall, spans)

    # Added: alignment top1/top3 accuracy statistics
    top1_correct = 0
    top1_total = 0
    top3_correct = 0
    top3_total = 0

    # Wrap the dataloader with tqdm for progress bar
    for step, batch in enumerate(tqdm(train_dataloader, desc="Evaluating", ncols=80)):
        # Get inputs
        code_inputs = batch[0].to(args.device)
        attn_mask = batch[1].to(args.device)
        position_idx = batch[2].to(args.device)
        nl_inputs = batch[3].to(args.device)
        ori2cur_pos = batch[4].to(args.device)
        match_list = batch[5]

        # Get batch roles
        batch_start_idx = step * code_inputs.size(0)
        batch_end_idx = min((step + 1) * code_inputs.size(0), len(roles))
        batch_roles = roles[batch_start_idx:batch_end_idx].to(args.device)

        with torch.inference_mode():
            outputs = model(code_inputs=code_inputs, attn_mask=attn_mask, position_idx=position_idx, nl_inputs=nl_inputs,
                role_indices=batch_roles)  # (batch_size, seq_length_code, hidden_size)
            code_hidden, nl_hidden = outputs.code_hidden.detach().cpu(), outputs.nl_hidden.detach().cpu()
            code_probs, nl_probs = outputs.code_scores.detach().cpu(), outputs.nl_scores.detach().cpu()

        batch_size = code_probs.size(0)
        for b in range(batch_size):
            # 1) Predicted sets using separate highlight layers
            pred_code = set((code_probs[b] > 0.5).nonzero(as_tuple=True)[0].tolist())
            pred_nl = set((nl_probs[b] > 0.4).nonzero(as_tuple=True)[0].tolist())
            
            # Use original hidden states (without role embeddings) for similarity computation
            # sim = torch.matmul(nl_hidden[b], code_hidden[b, :, :nl_hidden.size(-1)].transpose(0, 1))

            # 2) Ground-truth sets (flatten all spans)
            true_code: Set[int] = set()
            true_nl: Set[int] = set()
            align_precision_concepts: List[float] = []
            align_recall_concepts: List[float] = []
            
            # Track spans for this sample
            sample_spans = []
            sample_recalls = []

            valid_code_spans = batch[6][b]  # List[(step_desc, code_span)]
            step_centers = []
            step_token_sets = []
            for step_desc, code_span in valid_code_spans:
                indices = []
                for j in range(0, len(code_span), 2):
                    indices.extend(range(code_span[j], code_span[j+1]+1))
                if indices:
                    emb = code_hidden[b, indices].mean(dim=0)
                    step_centers.append(emb)
                    step_token_sets.append(set(indices))
            if step_centers:
                step_centers = torch.stack(step_centers)  # [num_steps, H]
            else:
                step_centers = torch.zeros((1, code_hidden.size(-1)))
            # print(f"  Step centers shape: {step_centers.shape}")
            # print(f"  Step token sets: {step_token_sets}")

            for comment_span, code_span in match_list[b]:
                nl_tokens = set()
                for j in range(0, len(comment_span), 2):
                    true_nl.update(range(comment_span[j], comment_span[j + 1] + 1))
                    nl_tokens.update(range(comment_span[j], comment_span[j + 1] + 1))

                for j in range(0, len(code_span), 2):
                    true_code.update(range(code_span[j], code_span[j + 1] + 1))

                # Calculate retrieval specific attention for this span
                span_tokens = set()
                for j in range(0, len(code_span), 2):
                    span_tokens.update(range(code_span[j], code_span[j + 1] + 1))
                inter = span_tokens & pred_code
                span_recall = 1.0 if len(inter) > 0 else 0.0
                
                sample_spans.append([(code_span[i], code_span[i+1]) for i in range(0, len(code_span), 2)])
                sample_recalls.append(span_recall)

                # --- New alignment evaluation method ---
                # For each comment token, compute similarity with all step centers
                for c in set(nl_tokens):
                    comment_emb = nl_hidden[b, c]  # [H]
                    if step_centers.shape[0] == 0:
                        continue
                    sims = torch.matmul(step_centers, comment_emb)  # [num_steps]
                    pred_steps = set((sims > 0.3).nonzero(as_tuple=True)[0].tolist())
                    # True label: which steps this comment token belongs to (i.e., which step's code span covers code_span)
                    true_steps = set()
                    for idx, token_set in enumerate(step_token_sets):
                        if len(token_set & span_tokens) > 0:
                            true_steps.add(idx)
                    # print(f"    Comment token idx: {c}")
                    # print(f"      Similarity to steps: {sims.tolist()}")
                    # print(f"      Pred steps: {pred_steps}")
                    # print(f"      True steps: {true_steps}")
                    _, align_precision, align_recall = jaccard_index(pred_steps, true_steps)
                    # print(f"      Precision: {align_precision}, Recall: {align_recall}")
                    align_precision_concepts.append(align_precision if align_precision is not None else 0)
                    align_recall_concepts.append(align_recall if align_recall is not None else 0)
                    # Added: alignment top1 accuracy
                    if step_centers.shape[0] > 0 and len(true_steps) > 0:
                        top1_idx = torch.argmax(sims).item()
                        if top1_idx in true_steps:
                            top1_correct += 1
                        top1_total += 1
                        # Added: alignment top3 accuracy
                        top3_indices = torch.topk(sims, min(3, step_centers.shape[0])).indices.tolist()
                        if any(idx in true_steps for idx in top3_indices):
                            top3_correct += 1
                        top3_total += 1

            # Calculate average recall for this sample
            sample_avg_recall = sum(sample_recalls) / len(sample_recalls) if sample_recalls else 0.0
            retrieval_recall_all.append(sample_avg_recall)
            # Store details for analysis
            retrieval_details_all.append((
                len(retrieval_recall_all) - 1,  # current index
                sample_avg_recall,
                sample_spans
            ))

            # Handle alignment metrics
            align_precision_all.append(np.mean(align_precision_concepts) if align_precision_concepts else 0.0)
            align_recall_all.append(np.mean(align_recall_concepts) if align_recall_concepts else 0.0)
            # print(np.mean(align_precision_concepts), np.mean(align_recall_concepts))

            ## Highlight performance
            jacc_nl,   precision_nl,   recall_nl   = jaccard_index(pred_nl,   true_nl)
            jacc_code, precision_code, recall_code = jaccard_index(pred_code, true_code)

            jacc_nl_all.append(jacc_nl if jacc_nl is not None else 0)
            precision_nl_all.append(precision_nl if precision_nl is not None else 0)
            recall_nl_all.append(recall_nl if recall_nl is not None else 0)

            jacc_code_all.append(jacc_code if jacc_code is not None else 0)
            precision_code_all.append(precision_code if precision_code is not None else 0)
            recall_code_all.append(recall_code if recall_code is not None else 0)

    # Sort details by recall for analysis
    retrieval_details_all.sort(key=lambda x: x[1])

    return (align_precision_all, align_recall_all), \
           (jacc_nl_all, precision_nl_all, recall_nl_all), \
           (jacc_code_all, precision_code_all, recall_code_all), \
           retrieval_details_all, \
           (top1_correct, top1_total), \
           (top3_correct, top3_total)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
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

    output_dir = os.path.join("your_path.pth")
    model.load_state_dict(torch.load(output_dir), strict=False)
    model.to(args.device)
    model.eval()

    # Load labeled samples first
    full_dataset = TextDataset(tokenizer, args, args.test_data_file, pool)

    # Create dataloader for labeled samples only
    train_sampler = SequentialSampler(full_dataset)
    train_dataloader = DataLoader(full_dataset,
                                  sampler=train_sampler,
                                  batch_size=1,
                                  num_workers=4,
                                  collate_fn=textdataset_collate_fn)

    (align_precision_all, align_recall_all), \
    (jacc_nl_all, precision_nl_all, recall_nl_all), \
    (jacc_code_all, precision_code_all, recall_code_all), \
    retrieval_details, \
    (top1_correct, top1_total), \
    (top3_correct, top3_total) = evaluate_metrics(train_dataloader, full_dataset, model, args, roles)

    # Create a dictionary with all metrics
    metrics_dict = {
        "alignment": {
            "precision": align_precision_all,
            "recall": align_recall_all,
            "top1_accuracy": top1_correct / top1_total if top1_total > 0 else 0.0,
            "top3_accuracy": top3_correct / top3_total if top3_total > 0 else 0.0
        },
        "nl_highlight": {
            "jaccard": jacc_nl_all,
            "precision": precision_nl_all, 
            "recall": recall_nl_all
        },
        "code_highlight": {
            "jaccard": jacc_code_all,
            "precision": precision_code_all,
            "recall": recall_code_all
        },
        "retrieval": {
            "details": retrieval_details
        }
    }

    # Save to JSON file
    metrics_path = os.path.join('your_dir', 'your_path.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nSaved evaluation metrics to {metrics_path}")

    # Print means
    print(f"Alignment Precision = {np.mean(align_precision_all)}, Alignment Recall = {np.mean(align_recall_all)}")
    print(f"Highlight Jaccard (NL) = {np.mean(jacc_nl_all)}, Precision (NL) = {np.mean(precision_nl_all)}, Recall (NL) = {np.mean(recall_nl_all)}")
    print(f"Highlight Jaccard (Code) = {np.mean(jacc_code_all)}, Precision (Code) = {np.mean(precision_code_all)}, Recall (Code) = {np.mean(recall_code_all)}")
    print(f"Retrieval Recall = {np.mean([detail[1] for detail in retrieval_details])}")
    print(f"Alignment Top 1 Accuracy = {top1_correct / top1_total if top1_total > 0 else 0.0}")
    print(f"Alignment Top 3 Accuracy = {top3_correct / top3_total if top3_total > 0 else 0.0}")

    def safe_f1(precision, recall, eps=1e-8):
        return 2 * precision * recall / (precision + recall + eps)

if __name__ == "__main__":
    main()