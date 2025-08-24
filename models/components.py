import torch
import torch.nn as nn
import torch.nn.functional as F

# --- RoPE Implementation ---
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = 0
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        if seq_len <= self.max_seq_len: return
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().float(), persistent=False)
        self.register_buffer("sin_cached", emb.sin().float(), persistent=False)

    def forward(self, seq_len, dtype):
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(dtype),
            self.sin_cached[:seq_len].to(dtype)
        )

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    # Add unsqueeze for correct broadcast over batch and head dimensions
    # (SeqLen, Dim) -> (1, 1, SeqLen, Dim) to match (B, H, S, Dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)

# --- Custom Attention Implementation (replaces nn.MultiheadAttention) ---
class CustomMHA(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(self, x, mask=None, rope_cos=None, rope_sin=None):
        B, S, D = x.shape
        # (B, S, D) -> (B, H, S, HeadDim)
        Q = self.Wq(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.Wk(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            Q = apply_rotary_pos_emb(Q, rope_cos, rope_sin)
            K = apply_rotary_pos_emb(K, rope_cos, rope_sin)

        # Prepare mask for SDPA (additive: 0 for attention, -inf for masking)
        attn_mask = None
        if mask is not None:
             attn_mask = torch.zeros_like(mask, dtype=Q.dtype, device=Q.device)
             # True in boolean mask means MASK
             attn_mask.masked_fill_(mask, float('-inf'))

        # Use optimized implementation (SDPA)
        context = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0
        )

        context = context.transpose(1, 2).contiguous().view(B, S, D)
        return self.Wo(context)

class MicroExpert(nn.Module):
    """
    Small Feed-Forward Network representing a single expert in the CSR layer.
    Structurally identical to FFN in standard Transformer.
    """
    def __init__(self, d_model, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "gelu":
            # Use 'tanh' approximation for GELU, often gives better results
            self.activation = nn.GELU(approximate='tanh')
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, x):
        # x shape: (Tokens_for_this_expert, d_model)
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class StandardFFN(MicroExpert):
    """
    Standard dense FFN used for Baseline model (when CSR is disabled).
    """
    # Inherits everything from MicroExpert
    pass

# We will use standard nn.MultiheadAttention from PyTorch.
# It is optimized and supports Causal Masking.

def create_causal_mask(seq_len, device):
    """
    Creates causal mask for decoder.
    In PyTorch MHA, True or 1 (for bool mask) means position should NOT be attended to.
    """
    # Upper triangle mask (without diagonal)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    return mask
