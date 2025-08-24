import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from einops import rearrange
from src.models.components import MicroExpert


class ChamberOfSemanticResonance(nn.Module):
    """
    The core innovation of SRA: A Sparse MoE layer routed via semantic similarity (Resonance).
    """

    def __init__(self, config):
        super().__init__()
        self.router_noise = getattr(config.csr, 'router_noise', 0.1)  # For stabilization

        self.d_model = config.d_model
        self.num_experts = config.csr.num_experts
        self.top_k = config.csr.top_k
        self.use_default_routing = getattr(config.csr, 'use_default_routing', False)
        if self.use_default_routing:
            # Default MoE routing uses a simple linear router to produce logits over experts
            self.router = nn.Linear(self.d_model, self.num_experts)

        # 1. Initialize Experts
        self.experts = nn.ModuleList([
            MicroExpert(
                self.d_model,
                config.d_ff,
                config.dropout,
                config.activation
            ) for _ in range(self.num_experts)
        ])

        # 2. Initialize Semantic Anchors (Learnable Parameters)
        self.semantic_anchors = nn.Parameter(torch.randn(self.num_experts, self.d_model))
        self._init_anchors(config.csr.anchor_init)
        # If using default learned gating, anchors are not used; freeze to avoid unnecessary grads
        if self.use_default_routing:
            self.semantic_anchors.requires_grad_(False)
            logging.info(
                f"CSR initialized with Learned Gating (linear router). num_experts={self.num_experts}, top_k={self.top_k}")
        else:
            logging.info(
                f"CSR initialized with Semantic Resonance routing (anchor cosine). num_experts={self.num_experts}, top_k={self.top_k}")

    def _init_anchors(self, strategy):
        """Initializes anchors to maximize initial dispersion."""
        if strategy == "orthogonal":
            # Orthogonal initialization is best for maximizing distance in space
            nn.init.orthogonal_(self.semantic_anchors)
            if self.num_experts > self.d_model:
                logging.warning(
                    "N_experts > D_model. Orthogonal initialization might not guarantee full orthogonality.")
        else:
            # Standard initialization (Kaiming Uniform)
            nn.init.kaiming_uniform_(self.semantic_anchors, a=math.sqrt(5))

    def _add_noise(self, scores):
        """Adds noise for stabilization during training (Noisy Top-K)."""
        if self.training and self.router_noise > 0:
            noise = torch.randn_like(scores) * self.router_noise
            return scores + noise
        return scores

    def forward(self, x):
        """
        Routes input tokens to experts based on semantic resonance.
        x shape: (Batch, SeqLen, D_model)
        """
        B, S, D = x.shape
        # Flatten batch and sequence dimensions: (B*S, D)
        x_flat = rearrange(x, 'b s d -> (b s) d')

        # --- Resonance Calculation (Cosine Similarity) ---
        # Perform normalization in FP32 for better precision and stability with AMP/BF16.
        # Epsilon (1e-8) is added to avoid division by zero.

        if self.use_default_routing:
            # Default routing: linear router logits
            # Align input dtype with router weights to avoid AMP bf16 vs fp32 mismatch
            x_r = x_flat.to(self.router.weight.dtype)
            resonance_scores = self.router(x_r)  # (B*S, N_experts)
            anchors_norm = None
        else:
            # CSR routing: cosine similarity to semantic anchors
            # (B*S, D)
            x_norm = F.normalize(x_flat.float(), p=2, dim=-1, eps=1e-8)
            # (N_experts, D)
            anchors_norm = F.normalize(self.semantic_anchors.float(), p=2, dim=-1, eps=1e-8)

            # (B*S, D) @ (D, N_experts) -> (B*S, N_experts)
            resonance_scores = torch.matmul(x_norm, anchors_norm.T)

        # Add noise for stabilization
        noisy_scores = self._add_noise(resonance_scores)

        # --- Top-K Gating ---
        # (B*S, TopK)
        topk_scores, topk_indices = torch.topk(noisy_scores, self.top_k, dim=-1)

        # Softmax to get routing weights. (Performed in FP32)
        gating_weights = F.softmax(topk_scores, dim=-1).to(x.dtype)  # Return to original dtype (e.g., bf16)

        # --- Efficient Expert Execution (Dispatch and Combine) ---

        # Initialize output tensor
        output_flat = torch.zeros_like(x_flat)

        # 1. Preparation for Dispatch
        # (B*S*K) - expert indices for each token
        expert_indices_flat = rearrange(topk_indices, 'bs k -> (bs k)')
        # (B*S*K) - corresponding weights
        weights_flat = rearrange(gating_weights, 'bs k -> (bs k)')

        # Create repeated input tensor for each of K choices
        # (B*S, D) -> (B*S, K, D) -> (B*S*K, D)
        k = topk_indices.size(1)
        x_expanded = rearrange(x_flat.unsqueeze(1).repeat(1, k, 1), 'bs k d -> (bs k) d')

        # (B*S*K) - Indices of original tokens (from 0 to B*S-1), repeated K times
        token_indices = torch.arange(B * S, device=x.device).repeat_interleave(k)

        # Group assignments by experts and iterate only over engaged experts
        # 1) Sort assignments by expert index to have contiguous segments
        order = torch.argsort(expert_indices_flat)
        sorted_experts = expert_indices_flat[order]
        sorted_token_idx = token_indices[order]
        sorted_weights = weights_flat[order]
        sorted_inputs = x_expanded[order]

        # 2) Find unique experts and their segment sizes
        unique_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)

        # 3) Iterate only over engaged experts; take corresponding data slice
        start = 0
        for e, cnt in zip(unique_experts.tolist(), counts.tolist()):
            end = start + cnt

            batch_inputs = sorted_inputs[start:end]
            batch_token_idx = sorted_token_idx[start:end]
            batch_weights = sorted_weights[start:end]

            # Computation by expert e for its batch
            expert_output = self.experts[e](batch_inputs)

            # Weight and accumulate to output
            weighted_output = expert_output * batch_weights.unsqueeze(-1)
            output_flat.index_add_(0, batch_token_idx, weighted_output)

            start = end

        # too much VRAM :(
        # E = self.num_experts
        # # Збираємо параметри експертів у тензори (E, ...)
        # W1 = torch.stack([self.experts[i].linear1.weight for i in range(E)], dim=0)  # (E, D_ff, D)
        # b1 = torch.stack([self.experts[i].linear1.bias   for i in range(E)], dim=0)  # (E, D_ff)
        # W2 = torch.stack([self.experts[i].linear2.weight for i in range(E)], dim=0)  # (E, D, D_ff)
        # b2 = torch.stack([self.experts[i].linear2.bias   for i in range(E)], dim=0)  # (E, D)
        #
        # # Вибираємо ваги відповідно до експерта для кожного токена
        # idx = expert_indices_flat
        # W1_sel = W1.index_select(0, idx)  # (N, D_ff, D)
        # b1_sel = b1.index_select(0, idx)  # (N, D_ff)
        # W2_sel = W2.index_select(0, idx)  # (N, D, D_ff)
        # b2_sel = b2.index_select(0, idx)  # (N, D)
        #
        # # Обчислення FFN для всіх токенів у паралелі
        # xN = x_expanded.to(W1_sel.dtype)              # (N, D)
        # # Лінійний 1: (N,1,D) @ (N,D,D_ff) -> (N,1,D_ff)
        # h = torch.bmm(xN.unsqueeze(1), W1_sel.transpose(1, 2)).squeeze(1) + b1_sel
        # # Активація та дропаут (беремо дропаут від першого експерта як спільний)
        # h = F.gelu(h)
        # h = self.experts[0].dropout(h)
        # # Лінійний 2: (N,1,D_ff) @ (N,D_ff,D) -> (N,1,D)
        # y = torch.bmm(h.unsqueeze(1), W2_sel.transpose(1, 2)).squeeze(1) + b2_sel
        # y = y.to(x.dtype)
        #
        # # Зважуємо та акумулюємо
        # weighted_output = y * weights_flat.unsqueeze(-1)
        # output_flat.index_add_(0, token_indices, weighted_output)

        # Return to original shape
        output = rearrange(output_flat, '(b s) d -> b s d', b=B, s=S)

        # Data for auxiliary losses
        # We return them in FP32 for accurate loss calculations, but convert back to model dtype for compatibility.
        aux_data = {
            # For Dispersion Loss
            "anchors_norm": anchors_norm.to(x.dtype),
            # For Load Balancing Loss (need all resonance_scores)
            "resonance_scores": resonance_scores.to(x.dtype),
            # For Utilization Analysis hooks (scripts/analyze.py)
            "topk_indices": topk_indices  # (B*S, TopK) int64
        }

        return output, aux_data
