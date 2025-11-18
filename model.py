import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import math
from sklearn.cluster import KMeans
from typing import Tuple, Optional, Union, List
import numpy as np
from torch import Tensor
from dataclasses import dataclass
from loss import FocalLoss, InfoNCELoss, CrossSampleContrastiveLoss
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers import AutoModel, AutoTokenizer
import logging

@dataclass
class OutputFeatures:
    code_ids: Optional[Tensor]  # list of code tokens shape [B, L_code,]
    code_hidden: Optional[Tensor]  # last hidden state [B, L_code, H]
    code_scores: Optional[Tensor]  # predicted highlight scores [B, L_code, ]
    code_indices: Optional[List[List[int]]] # predicted highlight potions, list of size B
    nl_ids: Optional[Tensor]  # list of comment tokens shape [B, L_nl,]
    nl_hidden: Optional[Tensor]  # last hidden state [B, L_nl, H]
    nl_scores: Optional[Tensor]  # predicted highlight scores [B, L_nl, ]
    nl_indices: Optional[List[List[int]]] # predicted highlight potions, list of size B
    sim_mat: Optional[Tensor] # similarity matrix [B, L_nl, L_code]

class Model(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_roles: int = 12,  # Number of different roles
        temperature: float = 0.2,
        gamma: float = 2.0,
        alpha: float = 0.75,
        highlight_threshold: float = 0.5,
        pool_k: int = 5,
        cross_sample_temp: float = 0.1,
        similarity_threshold: float = 0.3,
        use_cross_sample_loss: bool = True,
    ):
        """
        encoder: a HuggingFace-style transformer with .embeddings and .config.hidden_size
        num_roles: number of different roles to learn embeddings for
        temperature: softmax temperature for InfoNCE
        highlight_threshold: sigmoid cutoff for selecting important tokens in inference
        pool_k: number of top-k tokens to use in pooling
        cross_sample_temp: temperature for cross-sample contrastive loss
        similarity_threshold: threshold for filtering negative samples in cross-sample loss
        use_cross_sample_loss: whether to use cross-sample contrastive loss
        """
        super().__init__()
        self.encoder = encoder
        # Initialize learnable role embeddings with random weights
        self.role_embedding = nn.Embedding(num_roles, encoder.config.hidden_size)
        
        # Separate highlight layers for code and nl
        self.code_highlight_layer = nn.Linear(encoder.config.hidden_size, 1)  # With role embedding
        self.nl_highlight_layer = nn.Linear(encoder.config.hidden_size, 1)  # Without role embedding
        
        self.temperature = temperature
        self.gamma = gamma
        self.alpha = alpha
        self.highlight_threshold = highlight_threshold
        self.k = pool_k
        
        # Cross-sample contrastive learning components
        self.use_cross_sample_loss = use_cross_sample_loss
        if use_cross_sample_loss:
            # Initialize a natural language model for similarity calculation
            try:
                self.nl_model = AutoModel.from_pretrained('all-mpnet-base-v2-path')
                self.nl_tokenizer = AutoTokenizer.from_pretrained('all-mpnet-base-v2-path')
                # Freeze the NL model parameters
                for param in self.nl_model.parameters():
                    param.requires_grad = False
                print("Successfully loaded local all-mpnet-base-v2 for similarity calculation")
            except Exception as e:
                print(f"Warning: Could not load local all-mpnet-base-v2 model: {e}")
                try:
                    # Fallback to online model
                    self.nl_model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
                    self.nl_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
                    for param in self.nl_model.parameters():
                        param.requires_grad = False
                    print("Successfully loaded online all-mpnet-base-v2 as fallback")
                except Exception as e2:
                    print(f"Warning: Could not load any NL model for similarity calculation: {e2}")
                    self.nl_model = None
                    self.nl_tokenizer = None
            
            self.cross_sample_loss = CrossSampleContrastiveLoss(
                temperature=cross_sample_temp,
                similarity_threshold=similarity_threshold
            )
            
            # Add similarity cache for efficiency
            self.similarity_cache = {}

    def forward(self,
                code_inputs: Optional[Tensor] = None,
                attn_mask: Optional[Tensor] = None,
                position_idx: Optional[Tensor] = None,
                nl_inputs: Optional[Tensor] = None,
                role_indices: Optional[Tensor] = None,  # [batch_size, seq_length]
                force_return_tuple: bool = False
        ) -> Union[Tuple[Optional[BaseModelOutputWithPoolingAndCrossAttentions],
                         Optional[BaseModelOutputWithPoolingAndCrossAttentions]],
                   OutputFeatures]:

        code_outputs = None
        if code_inputs is not None:
            nodes_mask = position_idx.eq(0)
            token_mask = position_idx.ge(2)
            inputs_embeddings = self.encoder.embeddings.word_embeddings(code_inputs)
            nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
            nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
            avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
            inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]
            code_outputs = self.encoder(inputs_embeds=inputs_embeddings,
                                         attention_mask=attn_mask,
                                         position_ids=position_idx,
                                         output_attentions=False,
                                         output_hidden_states=True)

        nl_outputs = None
        if nl_inputs is not None:
            nl_outputs = self.encoder(nl_inputs,
                                     attention_mask=nl_inputs.ne(1),
                                     output_attentions=False,
                                     output_hidden_states=True)

        if self.training or force_return_tuple:
            return code_outputs, nl_outputs
        else:
            with torch.no_grad():
                code_hidden = code_probs = code_indices = None
                if code_inputs is not None:
                    code_hidden = code_outputs.last_hidden_state # [B, L, H]
                    code_hidden = F.normalize(code_hidden, p=2, dim=-1) # normalized hidden states # [B, L, H]
                    
                    # Add learnable role embeddings for code tokens
                    if role_indices is not None:
                        role_emb = self.role_embedding(role_indices)  # [B, L, H]
                        code_hidden_with_role = code_hidden + role_emb  # Add role embeddings to hidden state
                    else:
                        code_hidden_with_role = code_hidden
                    
                    # Calculate scores for all tokens
                    code_probs = torch.sigmoid(self.code_highlight_layer(code_hidden_with_role).squeeze(-1))  # [B, L]
                    
                    # Create mask for valid tokens (up to total_code_tokens)
                    valid_code_mask = torch.zeros_like(code_probs, dtype=torch.bool)
                    for b in range(code_inputs.size(0)):
                        code_tokens_2 = (code_inputs[b] == 2).nonzero().flatten()
                        if len(code_tokens_2) == 0:
                            total_code_tokens = 255 - 1
                        else:
                            total_code_tokens = int(code_tokens_2[0].item())
                        valid_code_mask[b, :total_code_tokens] = True
                    
                    # Zero out predictions for invalid tokens
                    code_probs = code_probs * valid_code_mask.float()
                    code_mask = code_probs > self.highlight_threshold
                    code_indices = [code_mask[b].nonzero(as_tuple=False).squeeze(-1).tolist() for b in range(code_mask.size(0))]

                nl_hidden = nl_probs = nl_indices = None
                if nl_inputs is not None:
                    nl_hidden = nl_outputs.last_hidden_state  # [B, L, H]
                    nl_hidden = F.normalize(nl_hidden, p=2, dim=-1)  # normalized hidden states # [B, L, H]
                    nl_probs = torch.sigmoid(self.nl_highlight_layer(nl_hidden).squeeze(-1))  # [B, L]
                    
                    # Create mask for valid tokens (up to total_comment_tokens)
                    valid_nl_mask = torch.zeros_like(nl_probs, dtype=torch.bool)
                    for b in range(nl_inputs.size(0)):
                        nl_tokens_2 = (nl_inputs[b] == 2).nonzero().flatten()
                        if len(nl_tokens_2) == 0:
                            total_comment_tokens = 127 - 1
                        else:
                            total_comment_tokens = int(nl_tokens_2[0].item())
                        valid_nl_mask[b, :total_comment_tokens] = True
                    
                    # Zero out predictions for invalid tokens
                    nl_probs = nl_probs * valid_nl_mask.float()
                    nl_mask = nl_probs > self.highlight_threshold-0.03
                    nl_indices = [nl_mask[b].nonzero(as_tuple=False).squeeze(-1).tolist() for b in range(nl_mask.size(0))]

                sim_mat = None
                if nl_inputs is not None and code_inputs is not None:
                    sim_mat = torch.bmm(nl_hidden, code_hidden.transpose(1, 2))  # always use `comment` as query [B, L_nl, L_code, H]

                outputs = OutputFeatures(
                    code_ids=code_inputs,
                    code_hidden=code_hidden,
                    code_scores=code_probs,
                    code_indices=code_indices,
                    nl_ids=nl_inputs,
                    nl_hidden=nl_hidden,
                    nl_scores=nl_probs,
                    nl_indices=nl_indices,
                    sim_mat=sim_mat
                )

            return outputs

    def pool(self, hidden: Tensor, scores: Tensor) -> Tensor:
        """
        Attention-weighted pooling of hidden states.
        """
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, L, 1]
        return (weights * hidden).sum(dim=1)  # [B, H]

    def retrieval_loss(
        self,
        code_outputs,
        nl_outputs,
    ) -> Tensor:
        # fixme: optional
        nl_h = nl_outputs.last_hidden_state    # [B, L, H]
        code_h = code_outputs.last_hidden_state
        nl_h = F.normalize(nl_h, p=2, dim=-1)
        code_h = F.normalize(code_h, p=2, dim=-1)

        nl_scores = self.nl_highlight_layer(nl_h).squeeze(-1)    # [B, L]
        code_scores = self.code_highlight_layer(code_h).squeeze(-1)

        nl_pooled = torch.tanh(self.pool(nl_h, nl_scores))    # [B, H]
        code_pooled = torch.tanh(self.pool(code_h, code_scores))

        sim_matrix = torch.matmul(nl_pooled, code_pooled.t())  # [B, B]
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        return F.cross_entropy(sim_matrix, labels)

    def set_tokenizer(self, tokenizer):
        """Set the tokenizer for debug printing. Must be called before training if debug info is needed."""
        self.tokenizer = tokenizer

    def compute_loss(
        self,
        code_inputs: Tensor,
        code_outputs: Tensor,
        nl_outputs: Tensor,
        sample_index: int,
        concept_matches: List[Tuple[List[int], List[int]]],
        total_code_tokens: int,
        total_comment_tokens: int,
        role_indices: Optional[Tensor] = None,  # [batch_size, seq_length]
        valid_code_spans: Optional[list] = None,  # New parameter for valid code spans
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # Extract and normalize last-layer hidden states for this example
        nl_hidden = nl_outputs.last_hidden_state[sample_index]   # [L_nl, H]
        code_hidden = code_outputs.last_hidden_state[sample_index]  # [L_code, H]
        nl_hidden = F.normalize(nl_hidden, p=2, dim=-1)
        code_hidden = F.normalize(code_hidden, p=2, dim=-1)

        # Add learnable role embeddings for code tokens
        if role_indices is not None:
            role_emb = self.role_embedding(role_indices[sample_index])  # [L, H]
            code_hidden_with_role = code_hidden + role_emb  # Add role embeddings to hidden state
        else:
            code_hidden_with_role = code_hidden

        # Highlight (token-level) loss using separate highlight layers
        nl_scores = self.nl_highlight_layer(nl_hidden).squeeze(-1)   # [L_nl]
        code_scores = self.code_highlight_layer(code_hidden_with_role).squeeze(-1) # [L_code]

        total_highlight_loss, nl_highlight_loss, code_highlight_loss = self.compute_highlight_loss(
            nl_scores, code_scores,
            concept_matches,
            total_code_tokens, total_comment_tokens
        )

        return total_highlight_loss, nl_highlight_loss, code_highlight_loss

    def compute_highlight_loss(
        self,
        nl_scores: Tensor,
        code_scores: Tensor,
        concept_matches: List[Tuple[List[int], List[int]]],
        total_code_tokens: int,
        total_comment_tokens: int
    ) -> Tuple[Tensor, Tensor, Tensor]:  # Return tuple of (total_loss, nl_loss, code_loss)

        # Create masks for valid tokens
        valid_nl_mask = torch.zeros(nl_scores.size(0), device=nl_scores.device, dtype=torch.bool)
        valid_code_mask = torch.zeros(code_scores.size(0), device=code_scores.device, dtype=torch.bool)
        valid_nl_mask[:total_comment_tokens] = True
        valid_code_mask[:total_code_tokens] = True

        # Build boolean masks for highlighted positions
        nl_mask   = torch.zeros(nl_scores.size(0),     device=nl_scores.device,   dtype=torch.bool) # [L_nl]
        code_mask = torch.zeros(code_scores.size(0),   device=code_scores.device, dtype=torch.bool) # [L_code]

        for c_span, k_span in concept_matches:
            # flatten comment spans
            c_inds = [
                i for j in range(0, len(c_span), 2)
                for i in range(c_span[j], c_span[j+1] + 1)
                if i <= total_comment_tokens
            ]
            # Debug: check if c_inds are out of bounds
            for idx in c_inds:
                if idx >= nl_scores.size(0):
                    print(f"[DEBUG] c_inds out of bounds: idx={idx}, nl_scores.size(0)={nl_scores.size(0)}, c_span={c_span}, total_comment_tokens={total_comment_tokens}")
            k_inds = [
                i for j in range(0, len(k_span), 2)
                for i in range(k_span[j], k_span[j+1] + 1)
                if i <= total_code_tokens
            ]
            # Debug: check if k_inds are out of bounds
            for idx in k_inds:
                if idx >= code_scores.size(0):
                    print(f"[DEBUG] k_inds out of bounds: idx={idx}, code_scores.size(0)={code_scores.size(0)}, k_span={k_span}, total_code_tokens={total_code_tokens}")
            nl_mask[c_inds]   = True
            code_mask[k_inds] = True

        # Only compute loss for valid tokens
        nl_mask = nl_mask & valid_nl_mask
        code_mask = code_mask & valid_code_mask

        # Use different FocalLoss parameters for nl and code
        loss_fn_nl = FocalLoss(gamma=self.gamma, alpha=0.5)
        loss_fn_code = FocalLoss(gamma=self.gamma, alpha=self.alpha)  # Hard-coded parameters for code highlight
        
        # Only compute loss for valid tokens
        nl_loss = loss_fn_nl(nl_scores[valid_nl_mask], nl_mask[valid_nl_mask].float())
        code_loss = loss_fn_code(code_scores[valid_code_mask], code_mask[valid_code_mask].float())

        # Calculate sparsity loss for both nl and code tokens
        neg_nl_mask = ~nl_mask & valid_nl_mask  # shape: [L_nl], bool
        # neg_code_mask = ~code_mask & valid_code_mask  # shape: [L_code], bool
        
        nl_probs = torch.sigmoid(nl_scores)  # [L_nl]
        # code_probs = torch.sigmoid(code_scores)  # [L_code]
        
        nl_sparsity_loss = nl_probs[neg_nl_mask].mean() if neg_nl_mask.any() else torch.tensor(0.0, device=nl_scores.device)
        # code_sparsity_loss = code_probs[neg_code_mask].mean() if neg_code_mask.any() else torch.tensor(0.0, device=code_scores.device)
        
        # total_highlight_loss = nl_loss + code_loss + 0.5 * nl_sparsity_loss + 0.5 * code_sparsity_loss
        total_highlight_loss = nl_loss + code_loss + 0.5 * nl_sparsity_loss

        return total_highlight_loss, nl_loss, code_loss

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two text strings using CLS token representations with caching."""
        if self.nl_model is None or self.nl_tokenizer is None:
            # Fallback to simple string similarity
            return 0  # Default similarity
        
        # Create cache key (sorted to ensure consistency regardless of order)
        cache_key = tuple(sorted([text1, text2]))
        
        # Check cache first
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
            
        try:
            # Tokenize and encode
            inputs = self.nl_tokenizer([text1, text2], 
                                     padding=True, 
                                     truncation=True, 
                                     max_length=512, 
                                     return_tensors='pt')
            
            # Move to same device as encoder
            device = next(self.encoder.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get embeddings using CLS token
            with torch.no_grad():
                outputs = self.nl_model(**inputs)
                # Use CLS token representations (first token)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [2, H]
                
            # Compute cosine similarity
            similarity = F.cosine_similarity(embeddings[0:1], embeddings[1:2], dim=1)
            similarity_value = similarity.item()
            
            # Cache the result
            self.similarity_cache[cache_key] = similarity_value
            
            # Limit cache size to prevent memory issues
            if len(self.similarity_cache) > 10000:
                # Remove oldest entries (simple FIFO)
                oldest_keys = list(self.similarity_cache.keys())[:1000]
                for key in oldest_keys:
                    del self.similarity_cache[key]
            
            return similarity_value
            
        except Exception as e:
            print(f"Error computing text similarity: {e}")
            return 0.5  # Default similarity

    def filter_negative_samples(
        self,
        comment_concepts_batch,  # List[List[str]]
        step_descriptions_batch, # List[List[str]]
        nl_hidden: torch.Tensor,  # [B, L_nl, H]
        code_hidden: torch.Tensor,  # [B, L_code, H]
        match_list: list,  # List[List[Tuple[List[int], List[int]]]]
        total_code_tokens_list: list,
        total_comment_tokens_list: list,
        valid_comment_spans_batch: list,
        valid_code_spans_batch: list,
        max_negative_samples_per_concept: int = 50
    ):
        """
        For each concept, negative pool = all code step centers in batch, excluding steps overlapping with the positive span.
        Negatives are selected by topK (emb_sim - nl_sim).
        """
        device = code_hidden.device
        # 1. Collect all code step centers and their token sets
        all_step_centers, all_step_descs, all_step_token_sets, all_step_batch_step = [], [], [], []
        # print("[filter_negative_samples] Collecting all code step centers...")
        for bidx, (code_spans, total_code_tokens) in enumerate(zip(valid_code_spans_batch, total_code_tokens_list)):
            for sidx, (step_desc, code_span) in enumerate(code_spans):
                indices = [i for j in range(0, len(code_span), 2) for i in range(code_span[j], code_span[j+1]+1) if i <= total_code_tokens]
                if not indices:
                    continue
                emb = code_hidden[bidx, torch.tensor(indices, device=device)].mean(dim=0)
                all_step_centers.append(emb)
                all_step_descs.append(step_desc)
                all_step_token_sets.append(set((bidx, i) for i in indices))
                all_step_batch_step.append((bidx, sidx))
        # print(f"[filter_negative_samples] Total step centers collected: {len(all_step_centers)}")
        if len(all_step_centers) == 0:
            # print("[filter_negative_samples] No step centers found, returning empty list.")
            return []
        all_step_centers = torch.stack(all_step_centers, dim=0)  # [M, H]

        # 2. Collect all concept descriptions
        all_concept_descs = []
        concept_desc_ptrs = []  # (bidx, cidx) -> idx in all_concept_descs
        for bidx, comment_spans in enumerate(valid_comment_spans_batch):
            for cidx, (desc, _) in enumerate(comment_spans):
                concept_desc_ptrs.append(len(all_concept_descs))
                all_concept_descs.append(desc)

        # 3. Batch all descriptions into NL encoder
        desc2vec = {}
        if self.nl_model is not None and self.nl_tokenizer is not None:
            unique_descs = list(set(all_concept_descs + all_step_descs))
            desc2idx = {d: i for i, d in enumerate(unique_descs)}
            inputs = self.nl_tokenizer(unique_descs, padding=True, truncation=True, max_length=128, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.nl_model(**inputs)
                desc_embeds = F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=-1)  # [N, H]
            for d, i in desc2idx.items():
                desc2vec[d] = desc_embeds[i]
        # else fallback: desc2vec is empty

        # 4. For each concept, exclude step centers covered by positive samples
        negative_sample_indices = []
        concept_desc_idx = 0
        for bidx, (match, comment_spans, total_comment_tokens) in enumerate(zip(match_list, valid_comment_spans_batch, total_comment_tokens_list)):
            batch_neg = []
            # print(f"  [Sample {bidx}] Concepts: {len(match)}")
            for cidx, (comment_span, code_span) in enumerate(match):
                pos_token_set = set((bidx, i) for j in range(0, len(code_span), 2) for i in range(code_span[j], code_span[j+1]+1))
                neg_indices = [i for i, step_set in enumerate(all_step_token_sets) if len(step_set & pos_token_set) == 0]
                # print(f"    [Concept {cidx}] pos_token_set size: {len(pos_token_set)}, neg_indices count: {len(neg_indices)}")
                if not neg_indices:
                    batch_neg.append([])
                    concept_desc_idx += 1
                    continue
                # anchor: concept center
                comment_indices = [i for j in range(0, len(comment_span), 2) for i in range(comment_span[j], comment_span[j+1]+1) if i <= total_comment_tokens]
                if not comment_indices:
                    batch_neg.append([])
                    concept_desc_idx += 1
                    continue
                anchor_emb = nl_hidden[bidx, torch.tensor(comment_indices, device=device)].mean(dim=0)
                concept_desc = all_concept_descs[concept_desc_idx]
                neg_embs = all_step_centers[neg_indices]
                neg_descs = [all_step_descs[i] for i in neg_indices]
                # emb_sim, nl_sim
                emb_sims = F.cosine_similarity(anchor_emb.unsqueeze(0), neg_embs, dim=-1)
                if desc2vec:
                    concept_desc_emb = desc2vec[concept_desc]
                    neg_desc_embs = torch.stack([desc2vec[d] for d in neg_descs], dim=0)
                    nl_sims = F.cosine_similarity(concept_desc_emb.unsqueeze(0), neg_desc_embs, dim=-1)
                else:
                    nl_sims = torch.tensor([self.compute_text_similarity(concept_desc, d) for d in neg_descs], device=anchor_emb.device)
                diff = emb_sims - nl_sims
                # print(f"    [Concept {cidx}] emb_sims shape: {emb_sims.shape}, nl_sims shape: {nl_sims.shape}, diff shape: {diff.shape}")
                topk = torch.topk(diff, k=min(max_negative_samples_per_concept, diff.size(0)), largest=True)
                topk_indices = [neg_indices[i.item()] for i in topk.indices]
                neg_batch_step = [all_step_batch_step[i] for i in topk_indices]
                # print(f"    [Concept {cidx}] Selected top-{len(topk_indices)} negative samples.")
                batch_neg.append(neg_batch_step)
                concept_desc_idx += 1
            negative_sample_indices.append(batch_neg)
        # print(f"[filter_negative_samples] Finished. negative_sample_indices shape: {len(negative_sample_indices)} (batch), {[len(x) for x in negative_sample_indices]} (concepts per sample)")
        return negative_sample_indices

    def compute_cross_sample_contrastive_loss_with_filtering(
        self,
        nl_hidden: Tensor,  # [B, L_nl, H]
        code_hidden: Tensor,  # [B, L_code, H]
        match_list: List[List[Tuple[List[int], List[int]]]],  # List of concept alignments for each sample
        total_code_tokens_list: List[int],  # List of total code tokens for each sample
        total_comment_tokens_list: List[int],  # List of total comment tokens for each sample
        valid_comment_spans_batch: List[List[Tuple[str, List[int]]]],  # List of (concept_text, comment_span) for each sample
        valid_code_spans_batch: List[List[Tuple[str, List[int]]]],  # List of (step_desc, code_span) for each sample
        similarity_threshold: float = 0.3,
        max_negative_samples_per_concept: int = 50,
        code_inputs: torch.Tensor = None,  # [B, L_code]
        nl_inputs: torch.Tensor = None,    # [B, L_nl]
    ) -> Tensor:
        """Compute cross-sample contrastive loss with filtered negative samples."""
        # print(f"\n=== Cross-Sample Contrastive Loss Debug ===")
        # print(f"Batch size: {len(valid_code_spans_batch)}")
        # print(f"Use cross-sample loss: {self.use_cross_sample_loss}")
        
        if not self.use_cross_sample_loss or len(valid_code_spans_batch) <= 1:
            return torch.tensor(0.0, device=nl_hidden.device)
        
        # Extract text from span tuples for similarity filtering
        comment_concepts_batch = []
        step_descriptions_batch = []
        
        for valid_comment_spans in valid_comment_spans_batch:
            comment_concepts = [concept_text for concept_text, _ in valid_comment_spans]
            comment_concepts_batch.append(comment_concepts)
        
        for valid_code_spans in valid_code_spans_batch:
            step_descriptions = [step_desc for step_desc, _ in valid_code_spans]
            step_descriptions_batch.append(step_descriptions)
        
        # print("Starting negative sample filtering...")
        # Filter negative samples based on semantic similarity
        negative_sample_indices = self.filter_negative_samples(
            comment_concepts_batch, step_descriptions_batch, nl_hidden, code_hidden, match_list,
            total_code_tokens_list, total_comment_tokens_list, valid_comment_spans_batch, valid_code_spans_batch, max_negative_samples_per_concept
        )
        # print(f"Negative sample filtering completed. Shape: {len(negative_sample_indices)}")
        # # Calculate total length at each level
        # level1_length = len(negative_sample_indices)  # Number of samples in batch
        # level2_length = sum(len(sample) for sample in negative_sample_indices)  # Total concepts across all samples
        # level3_length = sum(len(concept) for sample in negative_sample_indices for concept in sample)  # Total negative samples across all concepts
        
        # print(f"Level 1 (samples): {level1_length}")
        # print(f"Level 2 (concepts): {level2_length}")
        # print(f"Level 3 (negative samples): {level3_length}")
        # print(f"Total sum of all levels: {level1_length + level2_length + level3_length}")
        # print("Extracting concept representations...")
        # Extract concept representations
        comment_centers, code_centers, comment_to_code_map = self.extract_concept_representations(
            nl_hidden, code_hidden, match_list, 
            total_code_tokens_list, total_comment_tokens_list, valid_comment_spans_batch
        )
        # print(f"Extracted {len(comment_centers)} comment centers and {len(code_centers)} code centers")
        
        if len(comment_centers) == 0:
            return torch.tensor(0.0, device=nl_hidden.device)
        
        # print("Stacking tensors...")
        # Stack into tensors
        comment_centers = torch.stack(comment_centers)  # [N, H]
        code_centers = torch.stack(code_centers)  # [N, H]
        # print(f"Stacked shapes - comment_centers: {comment_centers.shape}, code_centers: {code_centers.shape}")
        
        # Use filtered negative samples
        all_code_centers = code_centers  # [N, H]
        # === DEBUG: Print actual positive and negative samples used (print specific string and tokenizer-recovered original text) ===
        if hasattr(self, 'tokenizer') and code_inputs is not None and nl_inputs is not None:
            try:
                for batch_idx, sample_negatives in enumerate(negative_sample_indices[:1]):
                    for concept_idx, concept_negatives in enumerate(sample_negatives[:5]):
                        # Positive sample
                        if batch_idx < len(valid_code_spans_batch) and concept_idx < len(match_list[batch_idx]):
                            if concept_idx < len(valid_code_spans_batch[batch_idx]):
                                pos_code_str, code_span = valid_code_spans_batch[batch_idx][concept_idx]
                                total_code_tokens = total_code_tokens_list[batch_idx]
                                if code_span is not None:
                                    code_indices = [i for j in range(0, len(code_span), 2) for i in range(code_span[j], code_span[j+1]+1) if i <= total_code_tokens]
                                    code_token_ids = code_inputs[batch_idx].detach().cpu().tolist()
                                    code_tokens = [code_token_ids[i] for i in code_indices]
                                    code_token_strs = self.tokenizer.convert_ids_to_tokens(code_tokens)
                                    code_raw_str = self.tokenizer.convert_tokens_to_string(code_token_strs)
                                    print(f"[DEBUG][CROSS] Positive code span idx={code_indices} tokens={' '.join(code_token_strs)} raw_str={code_raw_str} string={pos_code_str}")
                        # Negative samples
                        for neg_batch_idx, neg_step_idx in concept_negatives[:5]:
                            if neg_batch_idx < len(valid_code_spans_batch):
                                neg_valid_spans = valid_code_spans_batch[neg_batch_idx]
                                if neg_step_idx < len(neg_valid_spans):
                                    neg_code_str, neg_code_span = neg_valid_spans[neg_step_idx]
                                    total_tokens = total_code_tokens_list[neg_batch_idx]
                                    neg_code_indices = [i for j in range(0, len(neg_code_span), 2) for i in range(neg_code_span[j], neg_code_span[j+1]+1) if i <= total_tokens]
                                    neg_code_token_ids = code_inputs[neg_batch_idx].detach().cpu().tolist()
                                    neg_code_tokens = [neg_code_token_ids[i] for i in neg_code_indices]
                                    neg_code_token_strs = self.tokenizer.convert_ids_to_tokens(neg_code_tokens)
                                    neg_code_raw_str = self.tokenizer.convert_tokens_to_string(neg_code_token_strs)
                                    print(f"[DEBUG][CROSS] Negative code span (batch={neg_batch_idx}, step={neg_step_idx}) idx={neg_code_indices} tokens={' '.join(neg_code_token_strs)} raw_str={neg_code_raw_str} string={neg_code_str}")
            except Exception as e:
                print(f"[DEBUG][CROSS] Error printing cross-sample debug info: {e}")
        # === END DEBUG ===
        loss = self.cross_sample_loss(
            comment_centers, code_centers, all_code_centers, comment_to_code_map,
            negative_sample_indices, nl_hidden, code_hidden, total_code_tokens_list,
            valid_code_spans_batch, valid_comment_spans_batch, step_descriptions_batch
        )
        # print(f"Cross-sample loss computed: {loss.item():.6f}")
        # print("=== End Cross-Sample Debug ===\n")
        
        return loss

    def extract_concept_representations(
        self,
        nl_hidden: Tensor,  # [B, L_nl, H]
        code_hidden: Tensor,  # [B, L_code, H]
        concept_matches_batch: List[List[Tuple[List[int], List[int]]]],  # List of concept alignments for each sample
        total_code_tokens_list: List[int],  # List of total code tokens for each sample
        total_comment_tokens_list: List[int],  # List of total comment tokens for each sample
        valid_comment_spans_batch: List[List[Tuple[str, List[int]]]],  # List of (concept_text, comment_span) for each sample
    ) -> Tuple[List[Tensor], List[Tensor], List[int]]:
        """
        Extract concept representations from hidden states based on concept alignments.
        
        Returns:
            comment_centers: List of comment concept center tensors
            code_centers: List of corresponding code center tensors  
            comment_to_code_map: List mapping comment center index to code center index
        """
        # print(f"    Extracting concept representations for {len(concept_matches_batch)} samples")
        
        comment_centers = []
        code_centers = []
        comment_to_code_map = []
        
        for batch_idx in range(len(concept_matches_batch)):
            concept_matches = concept_matches_batch[batch_idx]
            valid_comment_spans = valid_comment_spans_batch[batch_idx]
            total_code_tokens = total_code_tokens_list[batch_idx]
            total_comment_tokens = total_comment_tokens_list[batch_idx]
            
            # print(f"      Sample {batch_idx}: {len(concept_matches)} concept matches, {len(valid_comment_spans)} valid comment spans")
            
            # Process each concept alignment
            for span_idx, (comment_span, code_span) in enumerate(concept_matches):
                # Extract comment indices
                comment_indices = [
                    i for j in range(0, len(comment_span), 2)
                    for i in range(comment_span[j], comment_span[j + 1] + 1)
                    if i <= total_comment_tokens
                ]
                
                # Extract code indices
                code_indices = [
                    i for j in range(0, len(code_span), 2)
                    for i in range(code_span[j], code_span[j + 1] + 1)
                    if i <= total_code_tokens
                ]
                
                if not comment_indices or not code_indices:
                    # print(f"        Skipping concept match {span_idx} - no valid indices")
                    continue
                
                # Get comment and code representations
                comment_emb = nl_hidden[batch_idx, comment_indices]  # [num_comment_tokens, H]
                code_emb = code_hidden[batch_idx, code_indices]  # [num_code_tokens, H]
                
                # Compute centers (mean of token representations)
                comment_center = comment_emb.mean(dim=0)  # [H]
                code_center = code_emb.mean(dim=0)  # [H]
                
                comment_centers.append(comment_center)
                code_centers.append(code_center)
                comment_to_code_map.append(len(code_centers) - 1)  # Map to current code center
                
                # print(f"        Concept match {span_idx}: {len(comment_indices)} comment tokens, {len(code_indices)} code tokens")
        
        # print(f"    Extracted {len(comment_centers)} concept pairs")
        return comment_centers, code_centers, comment_to_code_map