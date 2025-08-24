import math
import torch
import torch.nn as nn
from src.models.components import create_causal_mask, MicroExpert, RotaryPositionalEmbedding, CustomMHA
from src.models.csr import ChamberOfSemanticResonance

class SRABlock(nn.Module):
    """A single block of the SRA architecture."""
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model

        # Pre-LayerNorm configuration
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

        # Use CustomMHA
        self.attn = CustomMHA(
            d_model=self.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout
        )
        # (FFN/MoE)
        if config.csr.enabled:
            self.ff_or_moe = ChamberOfSemanticResonance(config)
        else:
            self.ff_or_moe = MicroExpert(self.d_model, config.d_ff, config.dropout, config.activation)
        self.use_csr = config.csr.enabled

    # Added rope_cos and rope_sin arguments
    def forward(self, x, mask=None, rope_cos=None, rope_sin=None):
        x_norm = self.ln1(x)
        # Pass RoPE embeddings
        attn_output = self.attn(x_norm, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)
        x = x + attn_output

        # 2. FFN/CSR (with residual connection)
        x_norm = self.ln2(x)

        if self.use_csr:
            ff_output, aux_data = self.ff_or_moe(x_norm)
        else:
            ff_output = self.ff_or_moe(x_norm)
            aux_data = {}

        x = x + ff_output

        return x, aux_data

class SRA(nn.Module):
    """The main Semantic Resonance Architecture model."""
    def __init__(self, model_config, vocab_size):
        super().__init__()
        self.config = model_config
        self.d_model = model_config.d_model
        self.vocab_size = vocab_size

        # 1. Token Embeddings
        self.token_embed = nn.Embedding(vocab_size, self.d_model)
        self.dropout = nn.Dropout(model_config.dropout)

        # 2. Initialize RoPE
        head_dim = self.d_model // model_config.n_heads
        self.rope = RotaryPositionalEmbedding(dim=head_dim, max_seq_len=model_config.max_seq_length)

        # 3. SRA Block Stack
        self.blocks = nn.ModuleList([SRABlock(model_config) for _ in range(model_config.n_layers)])

        # 4. Final LayerNorm and Head
        self.ln_f = nn.LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, vocab_size, bias=False)

        # Weight Tying
        if hasattr(self, 'token_embed') and self.token_embed.weight.shape == self.lm_head.weight.shape:
            self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)
        self._init_weights_gpt2_style(model_config.n_layers)

    def _init_weights(self, module):
         # (Standard initialization)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_weights_gpt2_style(self, n_layers):
        # Scale weights for layers added to residual stream (for stability)
        std = 0.02 / math.sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            # Wo in Attention and linear2 in Experts/FFN
            if name.endswith('Wo.weight') or name.endswith('linear2.weight'):
                 torch.nn.init.normal_(p, mean=0.0, std=std)

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        B, S = input_ids.shape

        # 1. Token Embedding
        x = self.token_embed(input_ids)
        x = self.dropout(x)

        # 2. Causal Mask and RoPE Generation
        mask = create_causal_mask(S, x.device)
        rope_cos, rope_sin = self.rope(S, dtype=x.dtype)

        # 3. Transformer Layers
        all_aux_data = []
        for block in self.blocks:
            x, aux_data = block(x, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)
            if aux_data:
                all_aux_data.append(aux_data)

        # 4. Final Processing
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Return aux_data list for loss calculation in Trainer
        return logits, all_aux_data
