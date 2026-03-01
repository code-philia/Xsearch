#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader


# =========================================================
# 0) Utility: URL -> cache path (md5)
# =========================================================
def code_url_to_cache_path(code_url: str, cache_dir: Path) -> Path:
    code_hash = hashlib.md5(code_url.encode("utf-8")).hexdigest()
    return cache_dir / f"{code_hash}.pt"


# =========================================================
# 1) Concept clustering for NL tokens (query side)
# =========================================================
def extract_and_cluster_concepts_nl(
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    highlight_threshold: float,
    cluster_threshold: float,
    device: torch.device,
):
    """
    Cluster important NL tokens into concept centroids using agglomerative clustering.

    Args:
        hidden_states: [L, D]
        probs:        [L]
    Returns:
        centroids: [C, D], weights: [C]
    """
    # IMPORTANT: This is query-side; number of important tokens is usually small,
    # so sklearn clustering is acceptable here.
    from sklearn.cluster import AgglomerativeClustering  # local import to avoid overhead if unused

    important_indices = sorted((probs > highlight_threshold).nonzero(as_tuple=True)[0].tolist())
    if not important_indices:
        return None, None

    vectors = hidden_states[important_indices]
    vectors = torch.nn.functional.normalize(vectors, dim=1)

    if len(important_indices) == 1:
        return vectors, probs[important_indices]

    vectors_np = vectors.detach().cpu().numpy()
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1.0 - cluster_threshold,
        )
        labels = clustering.fit_predict(vectors_np)
    except Exception:
        labels = np.zeros(len(vectors_np), dtype=np.int64)

    unique_labels = set(labels.tolist())
    centroids: List[torch.Tensor] = []
    weights: List[torch.Tensor] = []

    imp_idx_t = torch.tensor(important_indices, device=device, dtype=torch.long)

    for label in unique_labels:
        idx = np.where(labels == label)[0]  # local
        idx_t = torch.tensor(idx, device=device, dtype=torch.long)

        vecs = vectors[idx_t]  # [k, D]
        centroid = torch.nn.functional.normalize(vecs.mean(dim=0), dim=0)

        token_pos = imp_idx_t[idx_t]
        w = probs[token_pos].mean()

        centroids.append(centroid)
        weights.append(w)

    return torch.stack(centroids, dim=0), torch.stack(weights, dim=0)


# =========================================================
# 2) Query Dataset + collate (tokenizer singleton)
# =========================================================
class SimpleQDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def build_query_collate_fn(tokenizer: RobertaTokenizer, nl_length: int):
    def collate_fn(batch):
        nl_list = []
        urls = []
        for b in batch:
            doc = b.get("docstring")
            if not doc:
                tokens = b.get("docstring_tokens")
                doc = " ".join(tokens) if tokens else ""
            nl_list.append(doc)
            urls.append(b.get("url") or b.get("code_url") or "")
        nl_inputs = tokenizer(
            nl_list,
            padding=True,
            truncation=True,
            max_length=nl_length,
            return_tensors="pt",
        )
        return nl_inputs, urls, nl_list
    return collate_fn


# =========================================================
# 3) Fast code cache builder
#    - Use TextDataset + textdataset_collate_fn to provide position_idx / attn_mask
#    - Extract code centroids by top-k important tokens (no sklearn clustering)
#    - Optionally write per-code centroid files (md5.pt). Default: OFF (much faster)
# =========================================================
def build_packed_codebase_cache_fast(
    *,
    codebase_data_file: Path,
    roles_file: Path,
    packed_cache_path: Path,
    centroid_cache_dir: Path,
    tokenizer: RobertaTokenizer,
    model,
    device: torch.device,
    code_length: int,
    cache_batch_size: int,
    max_code_clusters: int,
    code_prob_threshold: float,
    no_per_code_cache: bool,
    pool_workers: int,
):
    """
    Build a packed codebase cache for fast retrieval scoring.

    Output file (torch.save tuple):
        packed_tensor: [N, Cmax, D] float32 on CPU
        packed_mask:   [N, Cmax]    float32 on CPU (1 for valid centroid slots)
        code_urls:     list[str] length N

    Centroid extraction (FAST path):
        - No sklearn clustering (CPU heavy)
        - Select top-k tokens by code_scores per sample
        - Only keep tokens whose score > code_prob_threshold
        - Normalize token vectors so dot-product behaves like cosine similarity

    Performance notes:
        - Uses batch-wise topk + gather to avoid per-sample loops and many small GPU->CPU copies.
        - One single GPU->CPU copy per batch for the [B, Cmax, D] centroid tensor.
        - By default we do NOT write per-code small cache files (no_per_code_cache=True).
    """
    import torch.nn.functional as F
    from dataloader import TextDataset, textdataset_collate_fn
    import multiprocessing

    centroid_cache_dir.mkdir(parents=True, exist_ok=True)
    packed_cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load role indices (must align with the codebase dataset order)
    roles_np = np.load(str(roles_file))
    roles = torch.from_numpy(roles_np).long()

    # Build a picklable args namespace for TextDataset
    ds_args = argparse.Namespace(
        nl_length=128,
        code_length=code_length,
        data_flow_length=0,
        lang="python",   # adjust if needed
        device=device,
    )

    # Build dataset (TextDataset internally uses multiprocessing pool.map during init)
    ctx = multiprocessing.get_context("fork")
    pool = ctx.Pool(pool_workers) if pool_workers > 0 else None
    try:
        codebase_dataset = TextDataset(tokenizer, ds_args, str(codebase_data_file), pool, compute_alignment=False)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # DataLoader: keep num_workers=0 to avoid nested multiprocessing issues
    codebase_dataloader = DataLoader(
        codebase_dataset,
        batch_size=cache_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=textdataset_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()

    # Probe hidden dim D from one forward pass
    with torch.inference_mode():
        sample_batch = next(iter(codebase_dataloader))
        code_inputs = sample_batch[0].to(device, non_blocking=True)
        attn_mask = sample_batch[1].to(device, non_blocking=True)
        position_idx = sample_batch[2].to(device, non_blocking=True)

        batch_roles = roles[: code_inputs.size(0)].to(device, non_blocking=True)

        out = model(
            code_inputs=code_inputs,
            attn_mask=attn_mask,
            position_idx=position_idx,
            role_indices=batch_roles,
        )
        if not hasattr(out, "code_hidden") or not hasattr(out, "code_scores"):
            keys = [k for k in dir(out) if not k.startswith("_")]
            raise RuntimeError(
                "Missing outputs.code_hidden / outputs.code_scores. "
                f"Available fields (partial): {keys[:60]}"
            )
        D = out.code_hidden.size(-1)

    N = len(codebase_dataset)
    Cmax = int(max_code_clusters)

    packed_tensor = torch.zeros((N, Cmax, D), dtype=torch.float32)  # CPU
    packed_mask = torch.zeros((N, Cmax), dtype=torch.float32)       # CPU
    code_urls: List[str] = [""] * N

    print(f"\n🧱 Cache missing → building codebase cache from: {codebase_data_file}")
    print(f"   Will save to: {packed_cache_path}")
    print(f"   Codebase size: {N}")
    print(f"   Per-code centroid cache dir: {centroid_cache_dir}")
    print(f"   Packed cache shape target: [N={N}, Cmax={Cmax}, D={D}]")
    if no_per_code_cache:
        print("   no_per_code_cache=True: will NOT write 40k+ small .pt files (faster).")
    else:
        print("   no_per_code_cache=False: will write per-code centroid .pt files (slower).")

    # Enable TF32 for faster matmul on Ampere+ (safe for inference)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Sample-level progress bar
    pbar = tqdm(total=N, desc="Building code cache (samples)")
    cursor = 0

    for batch in codebase_dataloader:
        # Unpack batch from textdataset_collate_fn
        code_inputs = batch[0].to(device, non_blocking=True)
        attn_mask = batch[1].to(device, non_blocking=True)
        position_idx = batch[2].to(device, non_blocking=True)

        B = code_inputs.size(0)
        batch_start = cursor
        batch_end = min(cursor + B, len(roles))
        batch_roles = roles[batch_start:batch_end].to(device, non_blocking=True)

        # Forward pass
        with torch.inference_mode():
            out = model(
                code_inputs=code_inputs,
                attn_mask=attn_mask,
                position_idx=position_idx,
                role_indices=batch_roles,
            )
            code_hidden = out.code_hidden  # [B, L, D]
            code_scores = out.code_scores  # [B, L]

        # ---- FAST batch centroid extraction ----
        # Mask scores below threshold, then take top-k per sample.
        # This avoids per-sample loops and many small GPU->CPU copies.
        B2, L, D2 = code_hidden.shape
        assert B2 == B and D2 == D

        scores_masked = code_scores.masked_fill(code_scores <= code_prob_threshold, -1e9)  # [B, L]
        topk_vals, topk_idx = torch.topk(scores_masked, k=Cmax, dim=1, largest=True)       # [B, Cmax]
        valid = topk_vals > -1e8                                                           # [B, Cmax]

        gathered = code_hidden.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, D))          # [B, Cmax, D]
        gathered = F.normalize(gathered, dim=2)                                             # normalize per centroid
        gathered = gathered * valid.unsqueeze(-1)                                           # zero-out invalid slots

        # Optional per-code cache:
        # Writing 40k+ small files is extremely slow; keep it disabled by default.
        # If you really want it, we write only for valid centroids.
        if not no_per_code_cache:
            # One-time CPU copies for saving per-code cache
            gathered_cpu_for_save = gathered.detach().cpu().float()
            valid_cpu_for_save = valid.detach().cpu()

            for i in range(B):
                idx_global = cursor + i
                ex = codebase_dataset.examples[idx_global]
                code_url = getattr(ex, "code_url", None) or getattr(ex, "url", None) or ""
                cache_fp = code_url_to_cache_path(code_url, centroid_cache_dir)

                k = int(valid_cpu_for_save[i].sum().item())
                cents_cpu = gathered_cpu_for_save[i, :k, :].contiguous() if k > 0 else torch.empty((0, D), dtype=torch.float32)
                torch.save(cents_cpu, str(cache_fp))

        # Single GPU->CPU copy per batch for packing
        gathered_cpu = gathered.detach().cpu().float()   # [B, Cmax, D]
        valid_cpu = valid.detach().cpu().float()         # [B, Cmax]

        packed_tensor[cursor:cursor + B, :, :] = gathered_cpu
        packed_mask[cursor:cursor + B, :] = valid_cpu

        # Fill URLs (lightweight Python loop)
        for i in range(B):
            idx_global = cursor + i
            ex = codebase_dataset.examples[idx_global]
            code_url = getattr(ex, "code_url", None) or getattr(ex, "url", None) or ""
            code_urls[idx_global] = code_url

        cursor += B
        pbar.update(B)

    pbar.close()

    # Save packed cache
    torch.save((packed_tensor, packed_mask, code_urls), str(packed_cache_path))
    print(f"\n✅ Cache built and saved: {packed_cache_path}")
    print(f"   packed_tensor: {tuple(packed_tensor.shape)}")
    print(f"   packed_mask:   {tuple(packed_mask.shape)}")
    print(f"   code_urls:     {len(code_urls)} items\n")


# =========================================================
# 4) Main: evaluate MRR with optional auto cache rebuild
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="XSearch evaluation with optional fast packed cache building")

    # Model / tokenizer
    parser.add_argument("--tokenizer_name", type=str, default="microsoft/graphcodebert-base")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/graphcodebert-base")
    parser.add_argument("--num_roles", type=int, default=20)

    # Required eval paths
    parser.add_argument("--packed_cache_file", type=str, required=True)
    parser.add_argument("--test_data_file", type=str, required=True)
    parser.add_argument("--xsearch_ckpt", type=str, required=True)

    # Auto cache rebuild inputs (required if cache missing or rebuild_cache)
    parser.add_argument("--codebase_data_file", type=str, default=None)
    parser.add_argument("--roles_file", type=str, default=None)

    # Lengths
    parser.add_argument("--nl_length", type=int, default=128)
    parser.add_argument("--code_length", type=int, default=256)

    # Batch sizes
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--cache_batch_size", type=int, default=64)

    # Device
    parser.add_argument("--device", type=str, default="cuda:0")

    # Query thresholds
    parser.add_argument("--highlight_threshold", type=float, default=0.4)
    parser.add_argument("--cluster_threshold", type=float, default=0.8)

    # Code cache thresholds
    parser.add_argument("--code_prob_threshold", type=float, default=0.4)
    parser.add_argument("--max_code_clusters", type=int, default=64)

    # Cache rebuild controls
    parser.add_argument("--rebuild_cache", action="store_true")

    # Per-code centroid cache directory
    parser.add_argument(
        "--centroid_cache_dir",
        type=str,
        default="preprocess_dataset/cache_code_centroids",
        help="Directory for per-code centroid cache files (md5.pt).",
    )
    parser.add_argument(
        "--no_per_code_cache",
        action="store_true",
        help="Do not write per-code centroid cache files. Only build packed cache (faster).",
    )

    # Multiprocessing workers for TextDataset conversion
    parser.add_argument("--pool_workers", type=int, default=16)

    # OOM fallback: keep code DB on CPU during scoring
    parser.add_argument("--code_on_cpu", action="store_true")

    args = parser.parse_args()

    # Locate XSearch root directory
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import Model
    try:
        from model import Model
    except Exception as e:
        raise RuntimeError(f"Cannot import model.py. Ensure evaluation.py is under XSearch/. Error: {e}")

    # Device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️ CUDA specified but not available. Switching to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"🚀 Device: {device}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Resolve path helper (relative to XSearch root)
    def resolve_path(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (project_root / pp)

    packed_cache_path = resolve_path(args.packed_cache_file)
    test_data_path = resolve_path(args.test_data_file)
    ckpt_path = resolve_path(args.xsearch_ckpt)
    codebase_path = resolve_path(args.codebase_data_file) if args.codebase_data_file else None
    roles_path = resolve_path(args.roles_file) if args.roles_file else None
    centroid_cache_dir = resolve_path(args.centroid_cache_dir)

    if not test_data_path.exists():
        raise FileNotFoundError(f"Test jsonl not found: {test_data_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load tokenizer / encoder / model
    print("🤖 Loading tokenizer & encoder...")
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path)

    model = Model(encoder, num_roles=args.num_roles)
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()

    # Auto rebuild cache if needed
    if args.rebuild_cache or (not packed_cache_path.exists()):
        if codebase_path is None or roles_path is None:
            raise RuntimeError(
                "Cache missing or rebuild requested, but --codebase_data_file / --roles_file not provided."
            )
        if not codebase_path.exists():
            raise FileNotFoundError(f"codebase_data_file not found: {codebase_path}")
        if not roles_path.exists():
            raise FileNotFoundError(f"roles_file not found: {roles_path}")

        build_packed_codebase_cache_fast(
            codebase_data_file=codebase_path,
            roles_file=roles_path,
            packed_cache_path=packed_cache_path,
            centroid_cache_dir=centroid_cache_dir,
            tokenizer=tokenizer,
            model=model,
            device=device,
            code_length=args.code_length,
            cache_batch_size=args.cache_batch_size,
            max_code_clusters=args.max_code_clusters,
            code_prob_threshold=args.code_prob_threshold,
            no_per_code_cache=args.no_per_code_cache,
            pool_workers=args.pool_workers,
        )

    # Load packed cache
    print(f"📦 Loading packed cache: {packed_cache_path}")
    packed_tensor, packed_mask, code_urls = torch.load(str(packed_cache_path), map_location="cpu")
    print(f"   Codebase Size: {packed_tensor.size(0)}")
    print(f"   packed_tensor shape: {tuple(packed_tensor.shape)}")
    print(f"   packed_mask   shape: {tuple(packed_mask.shape)}")

    url2idx = {u: i for i, u in enumerate(code_urls)}

    # Move code DB to desired device
    if args.code_on_cpu:
        code_db = packed_tensor                     # CPU
        code_mask = packed_mask.unsqueeze(-1)       # CPU
        print("⚠️ code_on_cpu=True: code_db stays on CPU (slower but safer).")
    else:
        code_db = packed_tensor.to(device)
        code_mask = packed_mask.to(device).unsqueeze(-1)

    # Read queries
    all_queries = []
    with test_data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_queries.append(json.loads(line))
    print(f"🧾 Queries Size: {len(all_queries)}")

    # Query dataloader
    loader = DataLoader(
        SimpleQDataset(all_queries),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=build_query_collate_fn(tokenizer, args.nl_length),
    )

    # Evaluate MRR
    all_rr: List[float] = []
    debug_printed = False

    print("🔍 Evaluating...")
    for (nl_inputs, q_urls, batch_raw_text) in tqdm(loader, total=len(loader), desc="Evaluating"):
        input_ids = nl_inputs.input_ids.to(device)

        with torch.inference_mode():
            outputs = model(nl_inputs=input_ids)
            if not hasattr(outputs, "nl_hidden") or not hasattr(outputs, "nl_scores"):
                keys = [k for k in dir(outputs) if not k.startswith("_")]
                raise RuntimeError(
                    "Cannot find outputs.nl_hidden / outputs.nl_scores.\n"
                    f"Available fields (partial): {keys[:60]}"
                )
            nl_hidden = outputs.nl_hidden  # [B, L, D]
            nl_scores = outputs.nl_scores  # [B, L]

        B = len(q_urls)
        for i in range(B):
            q_url = q_urls[i]

            q_cens, q_weights = extract_and_cluster_concepts_nl(
                nl_hidden[i],
                nl_scores[i],
                highlight_threshold=args.highlight_threshold,
                cluster_threshold=args.cluster_threshold,
                device=device,
            )

            if q_cens is None:
                all_rr.append(0.0)
                continue

            # Similarity scoring
            if args.code_on_cpu:
                q_cens_cpu = q_cens.detach().cpu()
                q_weights_cpu = q_weights.detach().cpu()

                # sim_matrix: [N, Cc, Cq]
                sim_matrix = torch.matmul(code_db, q_cens_cpu.T)
                sim_matrix = sim_matrix * code_mask + (1.0 - code_mask) * -1e9

                # max over code concepts: [N, Cq]
                max_sims = sim_matrix.max(dim=1).values

                # weighted sum over query concepts: [N]
                final_scores = (max_sims * q_weights_cpu.unsqueeze(0)).sum(dim=1)
                w_sum = q_weights_cpu.sum()
                if w_sum > 0:
                    final_scores = final_scores / w_sum
                scores_np = final_scores.numpy()
            else:
                q_cens = q_cens.to(device)
                q_weights = q_weights.to(device)

                sim_matrix = torch.matmul(code_db, q_cens.T)
                sim_matrix = sim_matrix * code_mask + (1.0 - code_mask) * -1e9
                max_sims = sim_matrix.max(dim=1).values
                final_scores = (max_sims * q_weights.unsqueeze(0)).sum(dim=1)
                w_sum = q_weights.sum()
                if w_sum > 0:
                    final_scores = final_scores / w_sum
                scores_np = final_scores.detach().cpu().numpy()

            gt_idx = url2idx.get(q_url, None)
            if gt_idx is None:
                all_rr.append(0.0)
                continue

            gt_score = scores_np[gt_idx]
            rank_val = int(np.sum(scores_np > gt_score) + 1)
            rr = 1.0 / rank_val
            all_rr.append(rr)

            if not debug_printed:
                print("\n[DEBUG CHECK] First Query Content:", batch_raw_text[i])
                print(f"              GT Score: {gt_score:.4f}, Rank: {rank_val}, RR: {rr:.4f}")
                debug_printed = True

    if len(all_rr) == 0:
        print("❌ No results collected.")
        return

    save_path = resolve_path("/your_ranks_path.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "count": len(all_rr),
        "ranks": all_rr,
    }

    with save_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"📝 Saved ranks JSON to: {save_path}")

    final_mrr = float(np.mean(all_rr))
    print(f"\n✅ Final MRR: {final_mrr:.6f} (Count: {len(all_rr)})")


if __name__ == "__main__":
    main()
