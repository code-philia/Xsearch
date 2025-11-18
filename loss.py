from torch import Tensor
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Tuple, List

class InfoNCELoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = "mean"
    ):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        anchor_emb: Tensor,
        pos_emb: Tensor,
        neg_emb: Tensor,
    ) -> Tensor:
        # cosine similarities for all positive pairs
        pos_similarities = F.cosine_similarity(
            anchor_emb.unsqueeze(1),  # [C, 1, H]
            pos_emb.unsqueeze(0),  # [1, P, H]
            dim=2  # → [C, P]
        )

        # pairwise similarity: [C, N]
        neg_similarities = F.cosine_similarity(
            anchor_emb.unsqueeze(1),  # [C, 1, H]
            neg_emb.unsqueeze(0),  # [1, N, H]
            dim=2
        )  # → [C, N]

        # InfoNCE loss
        sim_pos_exp = torch.exp(pos_similarities / self.temperature)  # [C, P]
        sim_neg_exp = torch.exp(neg_similarities / self.temperature)  # [C, N]

        all_score = sim_pos_exp.sum(dim=1, keepdim=True) + sim_neg_exp.sum(dim=1, keepdim=True) + 1e-8  # [C, 1]
        pos_score = sim_pos_exp.sum(dim=1, keepdim=True)  # [C, 1]

        loss = -torch.log(pos_score / all_score)  # [C, 1]

        all_sim_mean = pos_similarities.mean(dim=1) 
        coverage_penalty = 1.0 - all_sim_mean
        coverage_loss = coverage_penalty.mean()
        total_loss = loss.mean() + 0.5 * coverage_loss  # (1,)

        if self.reduction == "mean":
            return total_loss.mean()  # (1,)
        elif self.reduction == "sum":
            return total_loss.sum()
        else:
            return total_loss  # no reduction


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.9,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:

        # compute BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (L,)

        # prediction confidence (for ground-truth) = sigmoid * targets + (1-sigmoid) * (1-targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)  # (L,)
        mod_factor = (1 - p_t) ** self.gamma

        # alpha factor: for POS *alpha, for NEG *(1-alpha). The larger α is, the greater the gradient contribution of positive samples, and the model focuses more on improving the recall of a small number of positive samples.
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)  # (L,)
        loss = alpha_factor * mod_factor * bce_loss  # (L,)

        if self.reduction == "mean":
            return loss.mean()  # (1,)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # no reduction


class CrossSampleContrastiveLoss(nn.Module):
    """Cross-sample contrastive loss using cluster centers as representations."""
    
    def __init__(self, temperature: float = 0.1, similarity_threshold: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        
    def forward(
        self,
        comment_centers: Tensor,  # [N, H] - comment concept centers from batch
        code_centers: Tensor,     # [N, H] - corresponding code centers from batch
        all_code_centers: Tensor, # [M, H] - all code centers from batch (including non-corresponding)
        comment_to_code_map: List[int],  # [N] - maps comment center index to corresponding code center index
        negative_sample_indices: List[List[List[Tuple[int, int]]]],  # [batch_size][num_concepts][negative_indices]
        nl_hidden: Tensor,  # [B, L_nl, H] - comment hidden states
        code_hidden: Tensor,  # [B, L_code, H] - code hidden states
        total_code_tokens_list: List[int],  # List of total code tokens for each sample
        valid_code_spans_batch: List[List[Tuple[str, List[int]]]],  # List of (step_desc, code_span) for each sample
        valid_comment_spans_batch: List[List[Tuple[str, List[int]]]],  # List of (concept_text, comment_span) for each sample
        step_descriptions_batch: List[List[str]],  # Step descriptions for each sample
    ) -> Tensor:
        """
        Args:
            comment_centers: [N, H] - comment concept centers
            code_centers: [N, H] - corresponding code centers  
            all_code_centers: [M, H] - all code centers from batch
            comment_to_code_map: [N] - maps comment center index to corresponding code center index
            negative_sample_indices: [batch_size][num_concepts][negative_indices] - negative sample indices
            nl_hidden: [B, L_nl, H] - comment hidden states
            code_hidden: [B, L_code, H] - code hidden states
            total_code_tokens_list: List of total code tokens for each sample
            valid_code_spans_batch: List of (step_desc, code_span) for each sample
            valid_comment_spans_batch: List of (concept_text, comment_span) for each sample
            step_descriptions_batch: List of step descriptions for each sample
        """
        if len(comment_centers) == 0:
            return torch.tensor(0.0, device=comment_centers.device)
            
        # Normalize embeddings
        comment_centers = F.normalize(comment_centers, p=2, dim=-1)
        code_centers = F.normalize(code_centers, p=2, dim=-1)
        
        # Create a mapping from (batch_idx, concept_idx) to global comment center index
        concept_global_idx = 0
        batch_concept_to_global = {}
        for batch_idx, valid_spans in enumerate(valid_comment_spans_batch):
            for concept_idx in range(len(valid_spans)):
                batch_concept_to_global[(batch_idx, concept_idx)] = concept_global_idx
                concept_global_idx += 1
        
        # Compute loss for each comment concept
        concept_losses = []
        
        for batch_idx, sample_negatives in enumerate(negative_sample_indices):
            for concept_idx, concept_negatives in enumerate(sample_negatives):
                # Get the global index for this concept
                concept_key = (batch_idx, concept_idx)
                if concept_key not in batch_concept_to_global:
                    continue
                
                global_concept_idx = batch_concept_to_global[concept_key]
                if global_concept_idx >= len(comment_centers):
                    continue
                
                # Get the comment center for this concept
                comment_center = comment_centers[global_concept_idx]  # [H]
                
                # Get the corresponding code center
                code_center_idx = comment_to_code_map[global_concept_idx]
                if code_center_idx >= len(code_centers):
                    continue
                code_center = code_centers[code_center_idx]  # [H]
                
                # Build negative code centers for this concept
                negative_code_centers = []
                
                for neg_batch_idx, neg_step_idx in concept_negatives:
                    if neg_batch_idx >= len(valid_code_spans_batch):
                        continue
                    
                    # Get the valid code spans for this negative batch
                    neg_valid_spans = valid_code_spans_batch[neg_batch_idx]
                    
                    # Find the code spans corresponding to this step
                    if neg_step_idx < len(neg_valid_spans):
                        # Get the code spans for this step (step_desc, code_span)
                        _, code_span = neg_valid_spans[neg_step_idx]
                        
                        # Extract code indices from the span
                        total_tokens = total_code_tokens_list[neg_batch_idx]
                        code_indices = [
                            i for j in range(0, len(code_span), 2)
                            for i in range(code_span[j], code_span[j + 1] + 1)
                            if i <= total_tokens
                        ]
                        
                        if code_indices:
                            # Get the code representation for these tokens
                            code_repr = code_hidden[neg_batch_idx, code_indices].mean(dim=0)  # [H]
                            negative_code_centers.append(code_repr)
                
                if not negative_code_centers:
                    continue
                
                # Stack negative centers and normalize
                negative_code_centers = torch.stack(negative_code_centers)  # [K, H]
                negative_code_centers = F.normalize(negative_code_centers, p=2, dim=-1)
                
                # Combine positive and negative code centers
                all_centers = torch.cat([code_center.unsqueeze(0), negative_code_centers], dim=0)  # [1+K, H]
                
                # Calculate similarity between comment center and all centers
                sim_scores = torch.matmul(comment_center.unsqueeze(0), all_centers.t()).squeeze(0)  # [1+K]
                
                # Positive similarity is the first element (corresponding code center)
                pos_sim = sim_scores[0]  # scalar
                neg_sims = sim_scores[1:]  # [K]
                
                if len(neg_sims) == 0:
                    continue
                
                # InfoNCE loss for this concept
                pos_exp = torch.exp(pos_sim / self.temperature)
                neg_exp = torch.exp(neg_sims / self.temperature)
                
                numerator = pos_exp
                denominator = pos_exp + neg_exp.sum() + 1e-8
                
                concept_loss = -torch.log(numerator / denominator)
                concept_losses.append(concept_loss)
        
        if len(concept_losses) == 0:
            return torch.tensor(0.0, device=comment_centers.device)
        
        # Return average loss over all concepts
        return torch.stack(concept_losses).mean()
