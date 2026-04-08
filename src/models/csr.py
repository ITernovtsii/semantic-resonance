import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from einops import rearrange
from src.models.components import MicroExpert


class ChamberOfSemanticResonance(nn.Module):
    """
    Production-Ready SRA: A Sparse MoE layer routed via semantic resonance.
    Golden Configuration: Pure Cosine Routing + Global Learnable Temperature.

    Post-bugfix (Feb 13) + Code Freeze (Feb 17):
    - Softmax over ALL experts before top-k (fixes K=1 gradient starvation)
    - Scaled scores passed to aux_data (fixes load balancing blind spot)
    - Single global temperature (per-expert proved unnecessary)
    - No expert biases (collapsed without balance loss, near-zero with it)
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.num_experts = config.csr.num_experts
        self.top_k = config.csr.top_k

        # Routing type: 'cosine' (SRA) or 'linear' (Standard MoE baseline)
        rt = getattr(config.csr, 'routing_type', None)
        self.routing_type = rt if rt is not None else 'cosine'

        # Temperature scaling — mandatory for cosine routing
        ts = getattr(config.csr, 'temperature_scaling', None)
        self.temperature_scaling = ts if ts is not None else True

        # 1. Initialize Experts
        self.experts = nn.ModuleList([
            MicroExpert(self.d_model, config.d_ff, config.dropout, config.activation)
            for _ in range(self.num_experts)
        ])

        # 2. Initialize Routing Mechanism
        anchor_init = getattr(config.csr, 'anchor_init', None) or 'orthogonal'
        if self.routing_type == 'cosine':
            # SRA: Semantic Anchors — cosine similarity routing
            self.semantic_anchors = nn.Parameter(torch.randn(self.num_experts, self.d_model))
            if anchor_init == 'batch':
                # Placeholder random init — will be overwritten on first forward pass
                self._needs_batch_seed = True
            else:
                self._init_anchors(anchor_init)
                self._needs_batch_seed = False

            # Global Learnable Temperature (mandatory for cosine sharpness)
            if self.temperature_scaling:
                ti = getattr(config.csr, 'temperature_init', None)
                temp_init = float(ti) if ti is not None else 10.0
                self.temperature = nn.Parameter(torch.tensor(temp_init))
        else:
            # Standard MoE: learned linear gating (Switch Transformer style)
            self.gate = nn.Linear(self.d_model, self.num_experts, bias=False)
            nn.init.normal_(self.gate.weight, std=0.02)
            self._needs_batch_seed = False

    def _init_anchors(self, strategy):
        """Initializes anchors to maximize initial dispersion."""
        if strategy == "orthogonal":
            nn.init.orthogonal_(self.semantic_anchors)
            if self.num_experts > self.d_model:
                logging.warning(
                    "N_experts > D_model. Orthogonal init might not guarantee full orthogonality.")
        else:
            nn.init.kaiming_uniform_(self.semantic_anchors, a=math.sqrt(5))

    @torch.no_grad()
    def _seed_anchors_from_batch(self, x_flat, attention_mask=None):
        """Replace anchors with L2-normalized samples from actual hidden states.
        If attention_mask is provided, only sample from non-PAD tokens."""
        if attention_mask is not None:
            active = attention_mask.bool().view(-1)
            x_flat = x_flat[active]
        total_tokens = x_flat.size(0)
        rand_indices = torch.randperm(total_tokens, device=x_flat.device)[:self.num_experts]
        sampled = x_flat[rand_indices].clone().float()
        normalized = F.normalize(sampled, p=2, dim=-1)
        self.semantic_anchors.data.copy_(normalized)
        logging.info(f"CSR: Batch-seeded {self.num_experts} anchors from {total_tokens} tokens (PAD-filtered: {attention_mask is not None})")

    def set_top_k(self, new_top_k):
        """Dynamically update top_k for progressive scheduling."""
        if new_top_k != self.top_k:
            logging.info(f"CSR: Updating top_k: {self.top_k} → {new_top_k}")
            self.top_k = new_top_k

    def forward(self, x, attention_mask=None):
        """
        Routes input tokens to experts based on semantic resonance.
        x shape: (Batch, SeqLen, D_model)
        attention_mask: (Batch, SeqLen) — True for real tokens, False for PAD (optional)
        """
        B, S, D = x.shape
        x_flat = rearrange(x, 'b s d -> (b s) d')

        # Batch seeding: replace random anchors with real token embeddings (once)
        if self._needs_batch_seed:
            flat_mask = attention_mask.view(-1) if attention_mask is not None else None
            self._seed_anchors_from_batch(x_flat, flat_mask)
            self._needs_batch_seed = False

        # --- 1. Routing Score Calculation ---
        if self.routing_type == 'cosine':
            # Strict L2 Normalization (prevents magnitude collapse)
            x_norm = F.normalize(x_flat.float(), p=2, dim=-1, eps=1e-8)
            anchors_norm = F.normalize(self.semantic_anchors.float(), p=2, dim=-1, eps=1e-8)

            resonance_scores = torch.matmul(x_norm, anchors_norm.T)  # (B*S, E)

            # Apply absolute temperature (prevents gradient-spike inversion)
            if self.temperature_scaling:
                scaled_scores = resonance_scores * torch.abs(self.temperature)
            else:
                scaled_scores = resonance_scores
        else:
            # Linear gating (Standard MoE) — logits already unbounded
            scaled_scores = self.gate(x_flat.float())

        # --- 2. Full-Distribution Softmax ---
        # Crucial: Softmax over all experts BEFORE top-k ensures K=1 gradient flow
        routing_probs = F.softmax(scaled_scores, dim=-1)  # (B*S, E)

        # --- 3. Top-K Selection ---
        topk_probs, topk_indices = torch.topk(routing_probs, self.top_k, dim=-1)

        # Renormalize selected weights to sum to 1.0 (if K > 1)
        if self.top_k > 1:
            gating_weights = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            gating_weights = topk_probs

        gating_weights = gating_weights.to(x.dtype)

        # --- 4. Efficient Expert Execution (Dispatch and Combine) ---
        output_flat = torch.zeros_like(x_flat)

        expert_indices_flat = rearrange(topk_indices, 'bs k -> (bs k)')
        weights_flat = rearrange(gating_weights, 'bs k -> (bs k)')

        k = topk_indices.size(1)
        x_expanded = rearrange(x_flat.unsqueeze(1).repeat(1, k, 1), 'bs k d -> (bs k) d')
        token_indices = torch.arange(B * S, device=x.device).repeat_interleave(k)

        # Sort by expert index for contiguous segments
        order = torch.argsort(expert_indices_flat)
        sorted_experts = expert_indices_flat[order]
        sorted_token_idx = token_indices[order]
        sorted_weights = weights_flat[order]
        sorted_inputs = x_expanded[order]

        unique_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)

        start = 0
        for e, cnt in zip(unique_experts.tolist(), counts.tolist()):
            end = start + cnt
            batch_inputs = sorted_inputs[start:end]
            batch_token_idx = sorted_token_idx[start:end]
            batch_weights = sorted_weights[start:end]

            expert_output = self.experts[e](batch_inputs)
            weighted_output = expert_output * batch_weights.unsqueeze(-1)
            output_flat.index_add_(0, batch_token_idx, weighted_output)

            start = end

        output = rearrange(output_flat, '(b s) d -> b s d', b=B, s=S)

        # --- Auxiliary data for losses ---
        # Pass routing_probs directly (skip redundant softmax in losses.py)
        aux_data = {
            "routing_probs": routing_probs.to(x.dtype),
            "topk_indices": topk_indices,
        }

        return output, aux_data
