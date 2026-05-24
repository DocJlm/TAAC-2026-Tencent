"""PCVRHyFormer: A hybrid transformer model for post-click conversion rate prediction."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, NamedTuple, Tuple, Optional, Union


class ModelInput(NamedTuple):
    user_int_feats: torch.Tensor
    item_int_feats: torch.Tensor
    user_dense_feats: torch.Tensor
    item_dense_feats: torch.Tensor
    seq_data: dict        # {domain: tensor [B, S, L]}
    seq_lens: dict        # {domain: tensor [B]}
    seq_time_buckets: dict  # {domain: tensor [B, L]}
    timestamp: Optional[torch.Tensor] = None  # exposure timestamp, [B]


def _parse_fid_list(fid_list: Union[str, List[int], Tuple[int, ...], None]) -> List[int]:
    """Parse a comma-separated fid list while keeping train/infer configs portable."""
    if fid_list is None:
        return []
    if isinstance(fid_list, (list, tuple)):
        return [int(x) for x in fid_list]
    if not str(fid_list).strip():
        return []
    return [int(x.strip()) for x in str(fid_list).split(',') if x.strip()]


def _unpack_feature_spec(spec: Tuple[int, ...]) -> Tuple[int, int, int, int]:
    """Return (fid, vocab_size, offset, length) for both old and new specs."""
    if len(spec) == 4:
        fid, vs, offset, length = spec
    elif len(spec) == 3:
        vs, offset, length = spec
        fid = -1
    else:
        raise ValueError(f"Unexpected feature spec length: {len(spec)}")
    return int(fid), int(vs), int(offset), int(length)


def _feature_role(
    fid: int,
    vocab_size: int,
    length: int,
    pair_fids: List[int],
    low_card_threshold: int,
    high_card_threshold: int,
) -> str:
    if fid in pair_fids:
        return 'dense_aligned'
    if length > 1:
        return 'multi_hot'
    if vocab_size > high_card_threshold:
        return 'high_card'
    if 0 < vocab_size <= low_card_threshold:
        return 'low_card'
    return 'mid_card'


# ═══════════════════════════════════════════════════════════════════════════════
# Rotary Position Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════════════════


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin values.

    Attributes:
        dim: Rotary embedding dimension.
        max_seq_len: Maximum sequence length for cache.
        base: Base frequency for rotary encoding.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute inv_freq: (dim // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Precompute cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos().unsqueeze(0), persistent=False)  # (1, seq_len, dim)
        self.register_buffer('sin_cached', emb.sin().unsqueeze(0), persistent=False)  # (1, seq_len, dim)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for the given sequence length.

        Returns pre-computed slices from the cache. The cache is built once
        in __init__ with max_seq_len; no runtime expansion is performed so
        that the forward pass remains compatible with torch.compile().
        """
        cos = self.cos_cached[:, :seq_len, :].to(device)
        sin = self.sin_cached[:, :seq_len, :].to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps and negates the first and second halves of the last dimension."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_to_tensor(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Applies Rotary Position Embedding to a single tensor.

    Args:
        x: (B, num_heads, L, head_dim)
        cos: (1, L_max, head_dim) or (B, L, head_dim) for batch-specific positions.
        sin: Same shape as cos.

    Returns:
        Rotated tensor of shape (B, num_heads, L, head_dim).
    """
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)  # (*, 1, L, head_dim)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + rotate_half(x) * sin_


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Basic Components
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLU(nn.Module):
    """SwiGLU activation: x1 * SiLU(x2)."""

    def __init__(self, d_model: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden_dim = d_model * hidden_mult
        self.fc = nn.Linear(d_model, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * F.silu(x2)
        x = self.fc_out(x)
        return x


class RoPEMultiheadAttention(nn.Module):
    """Multi-head attention with Rotary Position Embedding support.

    Manually projects Q/K/V and reshapes for multi-head, then injects RoPE
    after projection and before dot-product. Uses F.scaled_dot_product_attention
    for efficient computation.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        rope_on_q: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_on_q = rope_on_q
        self.dropout = dropout

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)

        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        q_rope_cos: Optional[torch.Tensor] = None,
        q_rope_sin: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple:
        """Computes multi-head attention with optional RoPE.

        Args:
            query: (B, Lq, D)
            key: (B, Lk, D)
            value: (B, Lk, D)
            key_padding_mask: (B, Lk), True indicates padding positions.
            attn_mask: (Lq, Lk) or (B*num_heads, Lq, Lk), additive mask.
            attn_bias: Additive attention bias shaped (B, Lk), (B, Lq, Lk),
                or (B, 1, Lq, Lk). Positive values increase attention weight.
            rope_cos: (1, L, head_dim), RoPE for KV side (also used for Q
                unless q_rope_* is provided).
            rope_sin: Same shape as rope_cos.
            q_rope_cos: (B, Lq, head_dim) or (1, Lq, head_dim), Q-specific
                RoPE for cross-attention with gathered positions.
            q_rope_sin: Same shape as q_rope_cos.
            need_weights: Compatibility parameter, not used.

        Returns:
            Tuple of (output, None).
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # 1. Linear projection
        Q = self.W_q(query)  # (B, Lq, D)
        K = self.W_k(key)    # (B, Lk, D)
        V = self.W_v(value)  # (B, Lk, D)

        # 2. Reshape to (B, num_heads, L, head_dim)
        Q = Q.view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Apply RoPE independently to Q and K
        if rope_cos is not None and rope_sin is not None:
            # K always uses rope_cos/rope_sin (KV-side positional encoding)
            K = apply_rope_to_tensor(K, rope_cos, rope_sin)

            if self.rope_on_q:
                # Q side: prefer dedicated q_rope_cos/sin (top_k positions in LongerEncoder cross-attn)
                q_cos = q_rope_cos if q_rope_cos is not None else rope_cos
                q_sin = q_rope_sin if q_rope_sin is not None else rope_sin
                Q = apply_rope_to_tensor(Q, q_cos, q_sin)

        # 4. Convert padding / causal mask / learned bias to SDPA additive form.
        # A float additive mask is required once we add recency bias.
        sdpa_attn_mask = None
        neg_inf = torch.finfo(Q.dtype).min
        if attn_bias is not None:
            bias = attn_bias.to(dtype=Q.dtype, device=Q.device)
            if bias.dim() == 2:
                bias = bias.unsqueeze(1).unsqueeze(2).expand(B, 1, Lq, Lk)
            elif bias.dim() == 3:
                bias = bias.unsqueeze(1)
            elif bias.dim() != 4:
                raise ValueError(f"Unsupported attn_bias shape: {tuple(attn_bias.shape)}")
            sdpa_attn_mask = bias.expand(B, self.num_heads, Lq, Lk).clone()

        if key_padding_mask is not None:
            if sdpa_attn_mask is None:
                sdpa_attn_mask = Q.new_zeros(B, self.num_heads, Lq, Lk)
            pad = key_padding_mask.unsqueeze(1).unsqueeze(2).expand(B, self.num_heads, Lq, Lk)
            sdpa_attn_mask = sdpa_attn_mask.masked_fill(pad, neg_inf)

        if attn_mask is not None:
            mask = attn_mask.to(dtype=Q.dtype, device=Q.device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, Lq, Lk)
            elif mask.dim() == 3:
                mask = mask.view(B, self.num_heads, Lq, Lk)
            elif mask.dim() != 4:
                raise ValueError(f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}")
            if sdpa_attn_mask is None:
                sdpa_attn_mask = mask
            else:
                sdpa_attn_mask = sdpa_attn_mask + mask

        # 5. Scaled Dot-Product Attention
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=sdpa_attn_mask,
            dropout_p=dropout_p,
        )  # (B, num_heads, Lq, head_dim)

        # Replace NaN from all-padding softmax with 0 (zero vectors preserve original input via residual)
        out = torch.nan_to_num(out, nan=0.0)

        # 6. Reshape back and output projection
        out = out.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        G = self.W_g(query)
        out = out * torch.sigmoid(G)
        out = self.W_o(out)

        return out, None


class CrossAttention(nn.Module):
    """Cross-attention module.

    Query comes from global tokens (Q tokens), Key/Value comes from sequence
    tokens. Only applies RoPE to KV side (rope_on_q=False).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        ln_mode: str = 'pre'
    ) -> None:
        super().__init__()
        self.ln_mode = ln_mode

        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=False,
        )

        if ln_mode in ['pre', 'post']:
            self.norm_q = nn.LayerNorm(d_model)
            self.norm_kv = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes cross-attention between query tokens and sequence tokens.

        Args:
            query: (B, Nq, D), query tokens.
            key_value: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding positions.
            attn_bias: Optional additive attention bias for query-to-history reads.
            rope_cos: (1, L, head_dim), KV-side RoPE cosine values.
            rope_sin: (1, L, head_dim), KV-side RoPE sine values.

        Returns:
            Output tensor of shape (B, Nq, D).
        """
        residual = query

        if self.ln_mode == 'pre':
            query = self.norm_q(query)
            key_value = self.norm_kv(key_value)

        out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            attn_bias=attn_bias,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )

        out = residual + out

        if self.ln_mode == 'post':
            out = self.norm_q(out)

        return out


class RankMixerBlock(nn.Module):
    """HyFormer Query Boosting block.

    Performs three steps:
    1. Token Mixing: Parameter-free tensor reshaping.
    2. Per-token FFN: Shared-parameter feedforward network.
    3. Residual connection: Q_boost = Q + Q_e.

    Constraint: d_model must be divisible by n_total in 'full' mode.
    """

    def __init__(
        self,
        d_model: int,
        n_total: int,  # T = Nq + Nns
        hidden_mult: int = 4,
        dropout: float = 0.0,
        mode: str = 'full'  # 'full' | 'ffn_only' | 'none'
    ) -> None:
        super().__init__()
        self.T = n_total
        self.D = d_model
        self.mode = mode

        if mode == 'none':
            # Pure identity mapping, no submodules created
            return

        if mode == 'full':
            if d_model % n_total != 0:
                raise ValueError(
                    f"d_model={d_model} must be divisible by T={n_total} for token mixing."
                )
            self.d_sub = d_model // n_total

        # Per-token FFN (shared parameters) — used by both 'full' and 'ffn_only'
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)
        self.dropout = nn.Dropout(dropout)
        # Post-LN after residual to stabilize stacked block outputs
        self.post_norm = nn.LayerNorm(d_model)

    def token_mixing(self, Q: torch.Tensor) -> torch.Tensor:
        """Performs parameter-free token mixing via reshape and transpose.

        Steps:
        1. Splits channels into T subspaces: (B, T, D) -> (B, T, T, d_sub).
        2. Swaps token and subspace axes: (B, token, h, d_sub) -> (B, h, token, d_sub).
        3. Flattens back: (B, T, D).

        Args:
            Q: (B, T, D)

        Returns:
            Mixed tensor of shape (B, T, D).
        """
        B, T, D = Q.shape

        # (B, T, D) -> (B, T, T, d_sub)
        Q_split = Q.view(B, T, self.T, self.d_sub)

        # (B, token, h, d_sub) -> (B, h, token, d_sub)
        Q_rewired = Q_split.transpose(1, 2).contiguous()

        # (B, T, T, d_sub) -> (B, T, D)
        Q_hat = Q_rewired.view(B, T, D)
        return Q_hat

    def forward(self, Q: torch.Tensor) -> torch.Tensor:
        """Applies query boosting: token mixing, FFN, and residual connection.

        Args:
            Q: (B, T, D) where T = Nq + Nns.

        Returns:
            Boosted tensor of shape (B, T, D).
        """
        if self.mode == 'none':
            return Q

        # Token Mixing (parameter-free rewire) or identity
        if self.mode == 'full':
            Q_hat = self.token_mixing(Q)
        else:  # 'ffn_only'
            Q_hat = Q

        # Per-token FFN
        x = self.norm(Q_hat)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        Q_e = self.fc2(x)

        # Residual from original Q
        Q_boost = Q + Q_e
        Q_boost = self.post_norm(Q_boost)
        return Q_boost


class MultiSeqQueryGenerator(nn.Module):
    """Multi-sequence query generation module.

    Generates Q tokens independently for each sequence:
    For each sequence i:
        GlobalInfo_i = Concat(F1..FM, MeanPool(Seq_i))
        Q_i = [FFN_{i,1}(GlobalInfo_i), ..., FFN_{i,N}(GlobalInfo_i)]
    """

    def __init__(
        self,
        d_model: int,
        num_ns: int,
        num_queries: int,
        num_sequences: int,
        hidden_mult: int = 4,
        extra_context_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.num_sequences = num_sequences
        self.d_model = d_model
        self.extra_context_tokens = int(extra_context_tokens)

        global_info_dim = (num_ns + 1 + self.extra_context_tokens) * d_model

        # LayerNorm on global_info to prevent gradient explosion from large-dim concat
        self.global_info_norm = nn.LayerNorm(global_info_dim)

        # Each sequence has N independent FFNs
        self.query_ffns_per_seq = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(global_info_dim, d_model * hidden_mult),
                    nn.SiLU(),
                    nn.Linear(d_model * hidden_mult, d_model),
                    nn.LayerNorm(d_model),
                )
                for _ in range(num_queries)
            ])
            for _ in range(num_sequences)
        ])

    def forward(
        self,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        extra_context: Optional[torch.Tensor] = None,
    ) -> list:
        """Generates query tokens for each sequence.

        Args:
            ns_tokens: (B, M, D), shared NS tokens.
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S. True
                indicates padding.

        Returns:
            List of (B, Nq, D) query token tensors, length S.
        """
        B = ns_tokens.shape[0]
        ns_flat = ns_tokens.view(B, -1)  # (B, M*D)
        if self.extra_context_tokens > 0:
            if extra_context is None:
                extra_flat = ns_tokens.new_zeros(
                    B, self.extra_context_tokens * self.d_model)
            else:
                if extra_context.dim() == 2:
                    extra_context = extra_context.unsqueeze(1)
                extra_flat = extra_context.reshape(B, -1)
        else:
            extra_flat = None

        q_tokens_list = []
        for i in range(self.num_sequences):
            # MeanPool(Seq_i)
            valid_mask = ~seq_padding_masks[i]  # True = valid
            valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, L_i, 1)
            seq_sum = (seq_tokens_list[i] * valid_mask_expanded).sum(dim=1)  # (B, D)
            seq_count = valid_mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            seq_pooled = seq_sum / seq_count  # (B, D)

            # GlobalInfo_i = Concat(NS_flat, seq_pooled_i)
            if extra_flat is None:
                global_info = torch.cat([ns_flat, seq_pooled], dim=-1)
            else:
                global_info = torch.cat([ns_flat, extra_flat, seq_pooled], dim=-1)
            global_info = self.global_info_norm(global_info)

            # Generate N query tokens
            queries = [ffn(global_info) for ffn in self.query_ffns_per_seq[i]]
            q_tokens = torch.stack(queries, dim=1)  # (B, Nq, D)
            q_tokens_list.append(q_tokens)

        return q_tokens_list


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Encoders
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUEncoder(nn.Module):
    """Efficient attention-free sequence encoder.

    Structure: x + Dropout(SwiGLU(LN(x))).
    """

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.swiglu = SwiGLU(d_model, hidden_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Applies the SwiGLU encoder with residual connection.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding. Not used by
                this encoder variant.
            **kwargs: Absorbs rope_cos/rope_sin and other unused parameters.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        residual = x
        x = self.norm(x)
        x = self.swiglu(x)
        x = self.dropout(x)
        x = residual + x
        return x, key_padding_mask


class TransformerEncoder(nn.Module):
    """High-capacity sequence encoder with self-attention and RoPE.

    Structure: Standard Transformer Encoder Layer (Pre-LN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        hidden_mult: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Applies one Transformer encoder layer.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding positions.
            rope_cos: (1, L, head_dim), RoPE cosine values.
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            Tuple of (output tensor of shape (B, L, D), key_padding_mask).
        """
        # Self-Attention (Pre-LN) with RoPE
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = residual + x

        # FFN (Pre-LN)
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x, key_padding_mask

class LongerEncoder(nn.Module):
    """Top-K compressed sequence encoder.

    Adapts behavior based on input length:
    - L > top_k (first MultiSeqHyFormerBlock): Cross Attention.
      Q = latest top_k tokens, K/V = all seq tokens -> output (B, top_k, D).
    - L <= top_k (subsequent MultiSeqHyFormerBlocks): Self Attention.
      Q = K = V = top_k tokens -> output (B, top_k, D).

    Causal mask is only applied among top_k tokens (self-attention layers);
    the first cross-attention layer does not use a causal mask since Q and K
    have different lengths.

    Returns (output, new_key_padding_mask) so downstream can update the mask.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        top_k: int = 50,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        causal: bool = False
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.causal = causal

        # Pre-LN for attention
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        # Shared RoPEMHA for both cross and self attention
        self.attn = RoPEMultiheadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            rope_on_q=True,
        )

        # FFN (Pre-LN + residual)
        self.ffn_norm = nn.LayerNorm(d_model)
        hidden_dim = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout)
        )

    def _gather_top_k(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects the latest top_k valid tokens from each sample.

        Args:
            x: (B, L, D)
            key_padding_mask: (B, L), True indicates padding.

        Returns:
            top_k_tokens: (B, top_k, D)
            new_padding_mask: (B, top_k), True indicates padding.
            position_indices: (B, top_k), original position index for each
                selected token, used for Q-side RoPE.
        """
        B, L, D = x.shape
        device = x.device

        # Valid lengths per sample
        valid_len = (~key_padding_mask).sum(dim=1)  # (B,)

        # Start position for each sample: max(valid_len - top_k, 0)
        actual_k = torch.clamp(valid_len, max=self.top_k)  # (B,)
        start_pos = valid_len - actual_k  # (B,)

        # Build gather indices: (B, top_k)
        offsets = torch.arange(self.top_k, device=device).unsqueeze(0).expand(B, -1)  # (B, top_k)
        indices = start_pos.unsqueeze(1) + offsets  # (B, top_k)

        # For samples with valid_len < top_k, early indices may exceed valid range;
        # clamp to [0, L-1] and handle via mask below
        indices = torch.clamp(indices, min=0, max=L - 1)

        # Gather: (B, top_k, D)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, D)  # (B, top_k, D)
        top_k_tokens = torch.gather(x, dim=1, index=indices_expanded)

        # New padding mask: first (top_k - actual_k) positions are padding
        new_valid_len = actual_k  # (B,)
        pad_count = self.top_k - new_valid_len  # (B,)
        pos_indices = torch.arange(self.top_k, device=device).unsqueeze(0)  # (1, top_k)
        new_padding_mask = pos_indices < pad_count.unsqueeze(1)  # (B, top_k)

        # Zero out tokens at padding positions
        top_k_tokens = top_k_tokens * (~new_padding_mask).unsqueeze(-1).float()

        # position_indices for Q-side RoPE
        position_indices = indices  # (B, top_k)

        return top_k_tokens, new_padding_mask, position_indices

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the LongerEncoder with adaptive cross/self attention.

        Args:
            x: (B, L, D), sequence tokens.
            key_padding_mask: (B, L), True indicates padding.
            rope_cos: (1, L, head_dim), RoPE cosine values (length must cover
                original sequence length L).
            rope_sin: (1, L, head_dim), RoPE sine values.

        Returns:
            output: (B, top_k, D), compressed sequence.
            new_key_padding_mask: (B, top_k), updated padding mask.
        """
        B, L, D = x.shape

        if L > self.top_k:
            # === Cross Attention mode (first MultiSeqHyFormerBlock) ===
            # 1. Extract latest top_k tokens as query
            q, new_mask, q_pos_indices = self._gather_top_k(x, key_padding_mask)

            # 2. Pre-LN
            q_normed = self.norm_q(q)
            kv_normed = self.norm_kv(x)

            # 3. Build Q-side RoPE cos/sin by gathering from global cos/sin at top_k positions
            q_rope_cos = None
            q_rope_sin = None
            if rope_cos is not None and rope_sin is not None:
                # rope_cos: (1, L_max, head_dim), q_pos_indices: (B, top_k)
                head_dim = rope_cos.shape[2]
                # Expand to batch dimension
                cos_expanded = rope_cos.expand(B, -1, -1)  # (B, L_max, head_dim)
                sin_expanded = rope_sin.expand(B, -1, -1)
                idx = q_pos_indices.unsqueeze(-1).expand(-1, -1, head_dim)  # (B, top_k, head_dim)
                q_rope_cos = torch.gather(cos_expanded, 1, idx)  # (B, top_k, head_dim)
                q_rope_sin = torch.gather(sin_expanded, 1, idx)

            # 4. Cross Attention (no causal mask since Q and K have different lengths)
            attn_out, _ = self.attn(
                query=q_normed,
                key=kv_normed,
                value=kv_normed,
                key_padding_mask=key_padding_mask,  # Original (B, L) mask
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                q_rope_cos=q_rope_cos,
                q_rope_sin=q_rope_sin,
            )
            out = q + attn_out  # Residual based on q
        else:
            # === Self Attention mode (subsequent MultiSeqHyFormerBlocks) ===
            new_mask = key_padding_mask

            # Pre-LN (Q and KV share norm_q)
            x_normed = self.norm_q(x)

            # Causal mask
            attn_mask = None
            if self.causal:
                attn_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=x.device
                )

            attn_out, _ = self.attn(
                query=x_normed,
                key=x_normed,
                value=x_normed,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )
            out = x + attn_out

        # FFN (Pre-LN + residual)
        residual = out
        out = self.ffn_norm(out)
        out = self.ffn(out)
        out = residual + out

        return out, new_mask


def create_sequence_encoder(
    encoder_type: str,
    d_model: int,
    num_heads: int = 4,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    top_k: int = 50,
    causal: bool = False
) -> nn.Module:
    """Creates a sequence encoder of the specified type.

    Args:
        encoder_type: One of 'swiglu', 'transformer', or 'longer'.
        d_model: Model dimension.
        num_heads: Number of attention heads (used by transformer/longer).
        hidden_mult: FFN expansion multiplier.
        dropout: Dropout rate.
        top_k: Compression length for LongerEncoder (only used by longer).
        causal: Whether to use causal mask in LongerEncoder (only used by
            longer).

    Returns:
        A sequence encoder module.
    """
    if encoder_type == 'swiglu':
        return SwiGLUEncoder(d_model, hidden_mult, dropout)
    elif encoder_type == 'transformer':
        return TransformerEncoder(d_model, num_heads, hidden_mult, dropout)
    elif encoder_type == 'longer':
        return LongerEncoder(d_model, num_heads, top_k, hidden_mult, dropout, causal)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class MultiSeqHyFormerBlock(nn.Module):
    """Multi-sequence HyFormer block.

    Each of the S sequences independently performs Sequence Evolution and
    Query Decoding, then all Q tokens and shared NS tokens are merged for
    joint Query Boosting.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_queries: int,
        num_ns: int,
        num_sequences: int,
        seq_encoder_type: str = 'swiglu',
        hidden_mult: int = 4,
        dropout: float = 0.0,
        top_k: int = 50,
        causal: bool = False,
        rank_mixer_mode: str = 'full'
    ) -> None:
        super().__init__()
        self.num_sequences = num_sequences
        self.num_queries = num_queries
        self.num_ns = num_ns

        # Independent sequence encoder per sequence
        self.seq_encoders = nn.ModuleList([
            create_sequence_encoder(
                encoder_type=seq_encoder_type,
                d_model=d_model,
                num_heads=num_heads,
                hidden_mult=hidden_mult,
                dropout=dropout,
                top_k=top_k,
                causal=causal
            )
            for _ in range(num_sequences)
        ])

        # Independent cross-attention per sequence
        self.cross_attns = nn.ModuleList([
            CrossAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                ln_mode='pre'
            )
            for _ in range(num_sequences)
        ])

        # RankMixer: input token count = Nq * S + Nns
        n_total = num_queries * num_sequences + num_ns
        self.mixer = RankMixerBlock(
            d_model=d_model,
            n_total=n_total,
            hidden_mult=hidden_mult,
            dropout=dropout,
            mode=rank_mixer_mode
        )

    def forward(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_padding_masks: list,
        rope_cos_list: Optional[List[torch.Tensor]] = None,
        rope_sin_list: Optional[List[torch.Tensor]] = None,
        attn_bias_list: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> Tuple[list, torch.Tensor, list, list]:
        """Processes one multi-sequence HyFormer block step.

        Args:
            q_tokens_list: List of (B, Nq, D) tensors, length S.
            ns_tokens: (B, Nns, D)
            seq_tokens_list: List of (B, L_i, D) tensors, length S.
            seq_padding_masks: List of (B, L_i) masks, length S.
            rope_cos_list: List of (1, L_i, head_dim) tensors, length S.
            rope_sin_list: List of (1, L_i, head_dim) tensors, length S.
            attn_bias_list: Optional per-domain additive attention bias.

        Returns:
            A tuple (next_q_list, next_ns, next_seq_list, next_masks), where
            next_q_list is a list of (B, Nq, D) updated query tensors,
            next_ns is (B, Nns, D) updated non-sequence tokens,
            next_seq_list is a list of (B, L_i', D) encoded sequence tensors,
            and next_masks is a list of (B, L_i') updated padding masks.
        """
        S = self.num_sequences
        Nq = self.num_queries

        # 1. Independent Sequence Evolution per sequence
        next_seqs = []
        next_masks = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            result = self.seq_encoders[i](
                seq_tokens_list[i], seq_padding_masks[i],
                rope_cos=rc, rope_sin=rs,
            )
            next_seq_i, mask_i = result
            next_seqs.append(next_seq_i)
            next_masks.append(mask_i)

        # 2. Independent Query Decoding per sequence
        decoded_qs = []
        for i in range(S):
            rc = rope_cos_list[i] if rope_cos_list is not None else None
            rs = rope_sin_list[i] if rope_sin_list is not None else None
            attn_bias_i = None
            if attn_bias_list is not None:
                candidate = attn_bias_list[i]
                if candidate is not None and candidate.shape[-1] == next_seqs[i].shape[1]:
                    attn_bias_i = candidate
            decoded_q_i = self.cross_attns[i](
                q_tokens_list[i], next_seqs[i], next_masks[i],
                attn_bias=attn_bias_i,
                rope_cos=rc, rope_sin=rs,
            )
            decoded_qs.append(decoded_q_i)

        # 3. Token Fusion: concatenate all decoded_q + ns_tokens
        combined = torch.cat(decoded_qs + [ns_tokens], dim=1)  # (B, Nq*S + Nns, D)

        # 4. Query Boosting
        boosted = self.mixer(combined)  # (B, Nq*S + Nns, D)

        # 5. Split back into per-sequence Q and NS
        next_q_list = []
        offset = 0
        for i in range(S):
            next_q_list.append(boosted[:, offset:offset + Nq, :])
            offset += Nq
        next_ns = boosted[:, offset:, :]

        return next_q_list, next_ns, next_seqs, next_masks


# ═══════════════════════════════════════════════════════════════════════════════
# PCVRHyFormer Main Model
# ═══════════════════════════════════════════════════════════════════════════════


class GroupNSTokenizer(nn.Module):
    """NS tokenizer used by ns_tokenizer_type='group'.

    Groups discrete features by fid, applies shared embedding with mean
    pooling per multi-valued feature, then projects each group to a single
    NS token (one token per group).
    """

    def __init__(self, feature_specs: List[Tuple[int, int, int]],
                 groups: List[List[int]], emb_dim: int, d_model: int,
                 emb_skip_threshold: int = 0) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for spec in feature_specs:
            _, vs, _, _ = _unpack_feature_spec(spec)
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Per-group projection: num_fids_in_group * emb_dim -> d_model (with LayerNorm)
        self.group_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(group) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for group in groups
        ])

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds and projects grouped discrete features into NS tokens.

        Args:
            int_feats: (B, total_int_dim), concatenated integer features.

        Returns:
            Tokens of shape (B, num_groups, D).
        """
        tokens = []
        for group, proj in zip(self.groups, self.group_projs):
            fid_embs = []
            for fid_idx in group:
                _, vs, offset, length = _unpack_feature_spec(self.feature_specs[fid_idx])
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    # Filtered high-cardinality feature: output zero vector
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        # Single-value feature: direct lookup
                        fid_emb = emb_layer(int_feats[:, offset].long())  # (B, emb_dim)
                    else:
                        # Multi-value feature: lookup then mean pooling (ignoring padding=0)
                        vals = int_feats[:, offset:offset + length].long()  # (B, length)
                        emb_all = emb_layer(vals)  # (B, length, emb_dim)
                        mask = (vals != 0).float().unsqueeze(-1)  # (B, length, 1)
                        count = mask.sum(dim=1).clamp(min=1)  # (B, 1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count  # (B, emb_dim)
                fid_embs.append(fid_emb)
            cat_emb = torch.cat(fid_embs, dim=-1)  # (B, num_fids*emb_dim)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))  # (B, 1, D)
        return torch.cat(tokens, dim=1)  # (B, num_groups, D)


class RankMixerNSTokenizer(nn.Module):
    """NS Tokenizer following the RankMixer paper's approach.

    All group embedding vectors are concatenated into a single long vector,
    then equally split into num_ns_tokens segments, each projected to d_model.
    This allows num_ns_tokens to be chosen freely (independent of group count).
    """

    def __init__(
        self,
        feature_specs: List[Tuple[int, int, int]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
    ) -> None:
        """Initializes RankMixerNSTokenizer.

        Args:
            feature_specs: [(vocab_size, offset, length), ...] per feature.
            groups: List of feature index groups (defines semantic ordering).
            emb_dim: Embedding dimension per feature.
            d_model: Output token dimension.
            num_ns_tokens: Number of NS tokens to produce (T segments).
            emb_skip_threshold: Skip embedding for features with vocab > threshold.
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold

        # One embedding table per fid (None if skipped by emb_skip_threshold
        # or if vocab_size <= 0 / no vocab info).
        embs = []
        for spec in feature_specs:
            _, vs, _, _ = _unpack_feature_spec(spec)
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])
        # Map from fid index to position in self.embs (or -1 if filtered)
        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        # Compute total embedding dim: sum of all fids across all groups
        total_num_fids = sum(len(g) for g in groups)
        total_emb_dim = total_num_fids * emb_dim

        # Pad total_emb_dim to be divisible by num_ns_tokens
        self.chunk_dim = math.ceil(total_emb_dim / num_ns_tokens)
        self.padded_total_dim = self.chunk_dim * num_ns_tokens
        self._pad_size = self.padded_total_dim - total_emb_dim

        # Per-chunk projection: chunk_dim -> d_model with LayerNorm
        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.chunk_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(num_ns_tokens)
        ])

        logging.info(
            f"RankMixerNSTokenizer: {total_num_fids} fids, "
            f"total_emb_dim={total_emb_dim}, chunk_dim={self.chunk_dim}, "
            f"num_ns_tokens={num_ns_tokens}, pad={self._pad_size}"
        )

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        """Embeds all features, concatenates, splits, and projects.

        Args:
            int_feats: (B, total_int_dim) concatenated integer features.

        Returns:
            (B, num_ns_tokens, d_model) tensor.
        """
        # 1. Embed all fids in group order → flat cat
        all_embs = []
        for group in self.groups:
            for fid_idx in group:
                _, vs, offset, length = _unpack_feature_spec(self.feature_specs[fid_idx])
                emb_real_idx = self._emb_index[fid_idx]
                if emb_real_idx == -1:
                    fid_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim)
                else:
                    emb_layer = self.embs[emb_real_idx]
                    if length == 1:
                        fid_emb = emb_layer(int_feats[:, offset].long())
                    else:
                        vals = int_feats[:, offset:offset + length].long()
                        emb_all = emb_layer(vals)
                        mask = (vals != 0).float().unsqueeze(-1)
                        count = mask.sum(dim=1).clamp(min=1)
                        fid_emb = (emb_all * mask).sum(dim=1) / count
                all_embs.append(fid_emb)

        cat_emb = torch.cat(all_embs, dim=-1)  # (B, total_emb_dim)

        # 2. Pad if needed
        if self._pad_size > 0:
            cat_emb = F.pad(cat_emb, (0, self._pad_size))  # (B, padded_total_dim)

        # 3. Split into num_ns_tokens chunks and project each
        chunks = cat_emb.split(self.chunk_dim, dim=-1)  # list of (B, chunk_dim)
        tokens = []
        for chunk, proj in zip(chunks, self.token_projs):
            tokens.append(F.silu(proj(chunk)).unsqueeze(1))  # (B, 1, d_model)

        return torch.cat(tokens, dim=1)  # (B, num_ns_tokens, d_model)


class RoleStratifiedRankMixerNSTokenizer(nn.Module):
    """RankMixer tokenizer with deterministic role-aware feature assignment.

    The baseline RankMixer concatenates all sparse field embeddings in schema
    order and splits them into equal chunks. This variant keeps the same number
    of output tokens, but distributes feature roles across tokens so one token
    is not dominated by one role such as high-cardinality IDs or multi-hot
    fields.
    """

    _ROLE_ORDER = ('dense_aligned', 'low_card', 'multi_hot', 'mid_card', 'high_card')

    def __init__(
        self,
        feature_specs: List[Tuple[int, ...]],
        groups: List[List[int]],
        emb_dim: int,
        d_model: int,
        num_ns_tokens: int,
        emb_skip_threshold: int = 0,
        pair_fids: Union[str, List[int], Tuple[int, ...], None] = None,
        role_low_card_threshold: int = 1000,
        role_high_card_threshold: int = 100000,
    ) -> None:
        super().__init__()
        self.feature_specs = feature_specs
        self.groups = groups
        self.emb_dim = emb_dim
        self.num_ns_tokens = num_ns_tokens
        self.emb_skip_threshold = emb_skip_threshold
        self.pair_fids = _parse_fid_list(pair_fids)
        self.role_low_card_threshold = role_low_card_threshold
        self.role_high_card_threshold = role_high_card_threshold

        embs = []
        for spec in feature_specs:
            _, vs, _, _ = _unpack_feature_spec(spec)
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                embs.append(None)
            else:
                embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
        self.embs = nn.ModuleList([e for e in embs if e is not None])

        self._emb_index = []
        real_idx = 0
        for e in embs:
            if e is not None:
                self._emb_index.append(real_idx)
                real_idx += 1
            else:
                self._emb_index.append(-1)

        role_buckets: Dict[str, List[int]] = {role: [] for role in self._ROLE_ORDER}
        seen = set()
        for group in groups:
            for fid_idx in group:
                if fid_idx in seen:
                    continue
                seen.add(fid_idx)
                fid, vs, _, length = _unpack_feature_spec(feature_specs[fid_idx])
                role = _feature_role(
                    fid=fid,
                    vocab_size=vs,
                    length=length,
                    pair_fids=self.pair_fids,
                    low_card_threshold=role_low_card_threshold,
                    high_card_threshold=role_high_card_threshold,
                )
                role_buckets[role].append(fid_idx)

        token_features: List[List[int]] = [[] for _ in range(num_ns_tokens)]
        cursor = 0
        for role in self._ROLE_ORDER:
            ordered = sorted(
                role_buckets[role],
                key=lambda idx: (_unpack_feature_spec(feature_specs[idx])[0], idx),
            )
            for fid_idx in ordered:
                token_features[cursor % num_ns_tokens].append(fid_idx)
                cursor += 1
        self.token_feature_indices = token_features

        self.token_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(max(1, len(indices)) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            for indices in self.token_feature_indices
        ])

        role_counts = {role: len(role_buckets[role]) for role in self._ROLE_ORDER}
        logging.info(
            "RoleStratifiedRankMixerNSTokenizer: "
            f"tokens={num_ns_tokens}, role_counts={role_counts}, "
            f"token_sizes={[len(x) for x in self.token_feature_indices]}"
        )

    def _embed_feature(self, int_feats: torch.Tensor, fid_idx: int) -> torch.Tensor:
        _, _, offset, length = _unpack_feature_spec(self.feature_specs[fid_idx])
        emb_real_idx = self._emb_index[fid_idx]
        if emb_real_idx == -1:
            return int_feats.new_zeros(int_feats.shape[0], self.emb_dim, dtype=torch.float)
        emb_layer = self.embs[emb_real_idx]
        if length == 1:
            return emb_layer(int_feats[:, offset].long())
        vals = int_feats[:, offset:offset + length].long()
        emb_all = emb_layer(vals)
        mask = (vals != 0).float().unsqueeze(-1)
        count = mask.sum(dim=1).clamp(min=1)
        return (emb_all * mask).sum(dim=1) / count

    def forward(self, int_feats: torch.Tensor) -> torch.Tensor:
        cache: Dict[int, torch.Tensor] = {}
        tokens = []
        for indices, proj in zip(self.token_feature_indices, self.token_projs):
            if not indices:
                cat_emb = int_feats.new_zeros(int_feats.shape[0], self.emb_dim, dtype=torch.float)
            else:
                parts = []
                for fid_idx in indices:
                    if fid_idx not in cache:
                        cache[fid_idx] = self._embed_feature(int_feats, fid_idx)
                    parts.append(cache[fid_idx])
                cat_emb = torch.cat(parts, dim=-1)
            tokens.append(F.silu(proj(cat_emb)).unsqueeze(1))
        return torch.cat(tokens, dim=1)


class DenseIntPairCompressor(nn.Module):
    """Single-token dense projector with high-dense and aligned dense/int paths."""

    def __init__(
        self,
        user_dense_dim: int,
        user_dense_feature_specs: Optional[List[Tuple[int, int, int]]],
        user_int_feature_specs: List[Tuple[int, ...]],
        emb_dim: int,
        d_model: int,
        emb_skip_threshold: int = 0,
        pair_fids: Union[str, List[int], Tuple[int, ...], None] = None,
        high_dim_threshold: int = 128,
        residual_weight: float = 0.10,
    ) -> None:
        super().__init__()
        self.user_dense_dim = user_dense_dim
        self.emb_dim = emb_dim
        self.d_model = d_model
        self.residual_weight = float(residual_weight)
        self.pair_fids = _parse_fid_list(pair_fids)

        dense_specs = user_dense_feature_specs or []
        self.dense_meta = [(int(fid), int(offset), int(length)) for fid, offset, length in dense_specs]
        dense_by_fid = {fid: (offset, length) for fid, offset, length in self.dense_meta}
        int_by_fid = {}
        for idx, spec in enumerate(user_int_feature_specs):
            fid, vs, offset, length = _unpack_feature_spec(spec)
            int_by_fid[fid] = (idx, vs, offset, length)

        self.base_proj = nn.Sequential(
            nn.Linear(user_dense_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.semantic_slices = [
            (offset, length)
            for fid, offset, length in self.dense_meta
            if fid not in self.pair_fids and length >= high_dim_threshold
        ]
        self.scalar_slices = [
            (offset, length)
            for fid, offset, length in self.dense_meta
            if fid not in self.pair_fids and length < high_dim_threshold
        ]
        semantic_dim = sum(length for _, length in self.semantic_slices)
        scalar_dim = sum(length for _, length in self.scalar_slices)

        self.semantic_proj = nn.Sequential(
            nn.LayerNorm(max(1, semantic_dim)),
            nn.Linear(max(1, semantic_dim), d_model),
            nn.LayerNorm(d_model),
        )
        self.scalar_proj = nn.Sequential(
            nn.LayerNorm(max(1, scalar_dim)),
            nn.Linear(max(1, scalar_dim), d_model),
            nn.LayerNorm(d_model),
        )

        pair_specs = []
        pair_embs = []
        pair_gates = []
        for fid in self.pair_fids:
            if fid not in dense_by_fid or fid not in int_by_fid:
                continue
            dense_offset, dense_length = dense_by_fid[fid]
            _, vs, int_offset, int_length = int_by_fid[fid]
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            pair_specs.append((fid, dense_offset, dense_length, int_offset, int_length, skip))
            if skip:
                pair_embs.append(None)
            else:
                pair_embs.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            pair_gates.append(nn.Linear(1, emb_dim))
        self.pair_specs = pair_specs
        self.pair_embs = nn.ModuleList([e for e in pair_embs if e is not None])
        self.pair_gates = nn.ModuleList(pair_gates)
        self._pair_emb_index = []
        real_idx = 0
        for e in pair_embs:
            if e is None:
                self._pair_emb_index.append(-1)
            else:
                self._pair_emb_index.append(real_idx)
                real_idx += 1

        pair_dim = max(1, len(self.pair_specs)) * emb_dim
        self.pair_proj = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, d_model),
            nn.LayerNorm(d_model),
        )

        self.delta_proj = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.delta_proj[-1].weight)
        nn.init.zeros_(self.delta_proj[-1].bias)

        logging.info(
            "DenseIntPairCompressor: "
            f"semantic_dim={semantic_dim}, scalar_dim={scalar_dim}, "
            f"pairs={[p[0] for p in self.pair_specs]}, weight={self.residual_weight}"
        )

    def init_embeddings(self) -> None:
        for emb in self.pair_embs:
            nn.init.xavier_normal_(emb.weight.data)
            emb.weight.data[0, :] = 0

    def _slice_cat(self, dense_feats: torch.Tensor, slices: List[Tuple[int, int]]) -> torch.Tensor:
        if not slices:
            return dense_feats.new_zeros(dense_feats.shape[0], 1)
        return torch.cat([dense_feats[:, offset:offset + length] for offset, length in slices], dim=-1)

    def _pair_summary(self, dense_feats: torch.Tensor, int_feats: torch.Tensor) -> torch.Tensor:
        pair_vecs = []
        B = dense_feats.shape[0]
        for pair_idx, (_, d_off, d_len, i_off, i_len, skip) in enumerate(self.pair_specs):
            k = min(d_len, i_len)
            if k <= 0 or skip or self._pair_emb_index[pair_idx] == -1:
                pair_vecs.append(dense_feats.new_zeros(B, self.emb_dim))
                continue
            dense_slice = dense_feats[:, d_off:d_off + k].unsqueeze(-1)
            vals = int_feats[:, i_off:i_off + k].long()
            emb = self.pair_embs[self._pair_emb_index[pair_idx]](vals)
            gate = torch.sigmoid(self.pair_gates[pair_idx](dense_slice))
            mask = (vals != 0).float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1)
            pair_vecs.append((emb * gate * mask).sum(dim=1) / denom)
        if not pair_vecs:
            return dense_feats.new_zeros(B, self.emb_dim)
        return torch.cat(pair_vecs, dim=-1)

    def component_contexts(
        self,
        dense_feats: torch.Tensor,
        int_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return high-dim semantic, scalar/count, and dense-int pair summaries."""
        semantic = F.silu(self.semantic_proj(self._slice_cat(dense_feats, self.semantic_slices)))
        scalar = F.silu(self.scalar_proj(self._slice_cat(dense_feats, self.scalar_slices)))
        pair = F.silu(self.pair_proj(self._pair_summary(dense_feats, int_feats)))
        return semantic, scalar, pair

    def forward(self, dense_feats: torch.Tensor, int_feats: torch.Tensor) -> torch.Tensor:
        base = F.silu(self.base_proj(dense_feats))
        semantic, scalar, pair = self.component_contexts(dense_feats, int_feats)
        delta = self.delta_proj(torch.cat([semantic, scalar, pair], dim=-1))
        return base + self.residual_weight * delta


class LowRankCrossNet(nn.Module):
    """Low-rank DCN-style cross over stable user/item context tokens."""

    def __init__(self, d_model: int, rank: int = 16, layers: int = 2) -> None:
        super().__init__()
        self.input_dim = d_model * 3
        self.norm = nn.LayerNorm(self.input_dim)
        self.downs = nn.ModuleList([
            nn.Linear(self.input_dim, rank, bias=False)
            for _ in range(layers)
        ])
        self.ups = nn.ModuleList([
            nn.Linear(rank, self.input_dim, bias=True)
            for _ in range(layers)
        ])
        self.out = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(
        self,
        dense_ctx: torch.Tensor,
        item_ctx: torch.Tensor,
        user_ctx: torch.Tensor,
    ) -> torch.Tensor:
        x0 = self.norm(torch.cat([dense_ctx, item_ctx, user_ctx], dim=-1))
        x = x0
        for down, up in zip(self.downs, self.ups):
            crossed = up(F.gelu(down(x)))
            x = x + x0 * crossed
            x = self.norm(x)
        return self.out(x)


class TimeAttentionBias(nn.Module):
    """Small learned recency bias for query-to-sequence attention."""

    def __init__(self, num_time_buckets: int, hidden: int = 16, clip: float = 0.10) -> None:
        super().__init__()
        self.num_time_buckets = max(1, int(num_time_buckets))
        self.clip = float(clip)
        self.mlp = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        time_bucket_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L = time_bucket_ids.shape
        device = time_bucket_ids.device
        valid = (~padding_mask).float()
        valid_sum = valid.sum(dim=1, keepdim=True)
        denom = valid_sum.clamp(min=1.0)
        empty = (valid_sum <= 1.0e-6).float()

        bucket = time_bucket_ids.float() / float(max(self.num_time_buckets - 1, 1))
        pos = torch.arange(L, device=device, dtype=torch.float).unsqueeze(0).expand(B, -1)
        pos = pos / float(max(L - 1, 1))
        recency = pos * valid
        valid_len = valid_sum / float(max(L, 1))
        recent_k = min(64, L)
        recent_density = valid[:, -recent_k:].mean(dim=1, keepdim=True)

        feats = torch.stack([
            bucket,
            recency,
            valid_len.expand(B, L),
            recent_density.expand(B, L),
            empty.expand(B, L),
        ], dim=-1)
        bias = self.mlp(feats).squeeze(-1)
        bias = torch.tanh(bias) * self.clip
        return bias * valid


class QueryConditionedTimeAttentionBias(nn.Module):
    """Per-head time bias conditioned on the current query tokens."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_time_buckets: int,
        rank: int = 8,
        clip: float = 0.08,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.rank = max(1, int(rank))
        self.num_time_buckets = max(1, int(num_time_buckets))
        self.clip = float(clip)
        self.time_proj = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, self.rank),
            nn.SiLU(),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.num_heads * self.rank),
        )
        self.head_gate = nn.Parameter(torch.zeros(self.num_heads))

    def forward(
        self,
        q_tokens: torch.Tensor,
        time_bucket_ids: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L = time_bucket_ids.shape
        Nq = q_tokens.shape[1]
        device = time_bucket_ids.device
        valid = (~padding_mask).float()
        valid_sum = valid.sum(dim=1, keepdim=True)
        empty = (valid_sum <= 1.0e-6).float()

        bucket = time_bucket_ids.float() / float(max(self.num_time_buckets - 1, 1))
        pos = torch.arange(L, device=device, dtype=torch.float).unsqueeze(0).expand(B, -1)
        pos = pos / float(max(L - 1, 1))
        recency = pos * valid
        valid_len = valid_sum / float(max(L, 1))
        recent_k = min(64, L)
        recent_density = valid[:, -recent_k:].mean(dim=1, keepdim=True)

        feats = torch.stack([
            bucket,
            recency,
            valid_len.expand(B, L),
            recent_density.expand(B, L),
            empty.expand(B, L),
        ], dim=-1)
        time_low = self.time_proj(feats)  # (B, L, R)
        query_low = self.query_proj(q_tokens).view(B, Nq, self.num_heads, self.rank)
        raw = torch.einsum('bnhr,blr->bhnl', query_low, time_low) / math.sqrt(self.rank)
        bias = torch.tanh(raw) * self.clip
        bias = bias * self.head_gate.view(1, self.num_heads, 1, 1)
        return bias * valid.unsqueeze(1).unsqueeze(2)


class ExposureTimeContext(nn.Module):
    """User-side exposure-time context for query generation."""

    @staticmethod
    def feature_dim(multi_resolution: bool = False) -> int:
        return 20 if multi_resolution else 7

    def __init__(
        self,
        d_model: int,
        weight: float = 0.02,
        multi_resolution: bool = False,
        calendar_embeddings: bool = False,
        cross_calendar_time_context: bool = False,
        calendar_emb_dim: int = 4,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.multi_resolution = bool(multi_resolution)
        self.calendar_embeddings = bool(calendar_embeddings)
        self.cross_calendar_time_context = bool(cross_calendar_time_context)
        self.calendar_emb_dim = int(calendar_emb_dim)
        time_dim = self.feature_dim(self.multi_resolution)
        emb_dim = self.calendar_emb_dim * 6 if self.calendar_embeddings else 0
        if self.cross_calendar_time_context:
            emb_dim += self.calendar_emb_dim * 3
        self.input_dim = d_model * 2 + time_dim + emb_dim
        if self.calendar_embeddings:
            self.hour_emb = nn.Embedding(24, self.calendar_emb_dim)
            self.hour_cn_emb = nn.Embedding(24, self.calendar_emb_dim)
            self.weekday_emb = nn.Embedding(7, self.calendar_emb_dim)
            self.weekday_cn_emb = nn.Embedding(7, self.calendar_emb_dim)
            self.period3_emb = nn.Embedding(8, self.calendar_emb_dim)
            self.period6_emb = nn.Embedding(4, self.calendar_emb_dim)
            for emb in [
                self.hour_emb, self.hour_cn_emb, self.weekday_emb,
                self.weekday_cn_emb, self.period3_emb, self.period6_emb,
            ]:
                nn.init.xavier_normal_(emb.weight.data)
        if self.cross_calendar_time_context:
            self.hour_weekday_emb = nn.Embedding(24 * 7, self.calendar_emb_dim)
            self.period3_weekday_emb = nn.Embedding(8 * 7, self.calendar_emb_dim)
            self.weekend_hour_emb = nn.Embedding(2 * 24, self.calendar_emb_dim)
            for emb in [
                self.hour_weekday_emb,
                self.period3_weekday_emb,
                self.weekend_hour_emb,
            ]:
                nn.init.xavier_normal_(emb.weight.data)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    @staticmethod
    def build_time_features(
        timestamp: Optional[torch.Tensor],
        ref: torch.Tensor,
        multi_resolution: bool = False,
    ) -> torch.Tensor:
        B = ref.shape[0]
        time_dim = ExposureTimeContext.feature_dim(multi_resolution)
        if timestamp is None:
            return ref.new_zeros(B, time_dim)
        ts = timestamp.to(device=ref.device, dtype=torch.float32)
        hour = torch.remainder(torch.floor(ts / 3600.0), 24.0)
        weekday = torch.remainder(torch.floor(ts / 86400.0) + 4.0, 7.0)
        hour_angle = hour * (2.0 * math.pi / 24.0)
        weekday_angle = weekday * (2.0 * math.pi / 7.0)
        weekend = ((weekday >= 5.0).float())
        if multi_resolution:
            ts_cn = ts + 8.0 * 3600.0
            hour_cn = torch.remainder(torch.floor(ts_cn / 3600.0), 24.0)
            weekday_cn = torch.remainder(torch.floor(ts_cn / 86400.0) + 4.0, 7.0)
            hour_cn_angle = hour_cn * (2.0 * math.pi / 24.0)
            weekday_cn_angle = weekday_cn * (2.0 * math.pi / 7.0)
            period3 = torch.floor(hour_cn / 3.0)
            period6 = torch.floor(hour_cn / 6.0)
            period3_angle = period3 * (2.0 * math.pi / 8.0)
            period6_angle = period6 * (2.0 * math.pi / 4.0)
            weekend_cn = ((weekday_cn >= 5.0).float())
            return torch.stack([
                torch.sin(hour_angle),
                torch.cos(hour_angle),
                torch.sin(hour_cn_angle),
                torch.cos(hour_cn_angle),
                torch.sin(weekday_angle),
                torch.cos(weekday_angle),
                torch.sin(weekday_cn_angle),
                torch.cos(weekday_cn_angle),
                torch.sin(period3_angle),
                torch.cos(period3_angle),
                torch.sin(period6_angle),
                torch.cos(period6_angle),
                weekend,
                weekend_cn,
                hour / 23.0,
                hour_cn / 23.0,
                weekday / 6.0,
                weekday_cn / 6.0,
                period3 / 7.0,
                period6 / 3.0,
            ], dim=-1)
        return torch.stack([
            torch.sin(hour_angle),
            torch.cos(hour_angle),
            torch.sin(weekday_angle),
            torch.cos(weekday_angle),
            weekend,
            hour / 23.0,
            weekday / 6.0,
        ], dim=-1)

    @staticmethod
    def build_time_indices(
        timestamp: Optional[torch.Tensor],
        ref: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if timestamp is None:
            return None
        ts = timestamp.to(device=ref.device, dtype=torch.float32)
        ts_cn = ts + 8.0 * 3600.0
        hour = torch.remainder(torch.floor(ts / 3600.0), 24.0).long()
        hour_cn = torch.remainder(torch.floor(ts_cn / 3600.0), 24.0).long()
        weekday = torch.remainder(torch.floor(ts / 86400.0) + 4.0, 7.0).long()
        weekday_cn = torch.remainder(torch.floor(ts_cn / 86400.0) + 4.0, 7.0).long()
        period3 = torch.floor(hour_cn.float() / 3.0).long().clamp(0, 7)
        period6 = torch.floor(hour_cn.float() / 6.0).long().clamp(0, 3)
        return hour, hour_cn, weekday, weekday_cn, period3, period6

    def _calendar_embedding_features(
        self,
        timestamp: Optional[torch.Tensor],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        B = ref.shape[0]
        if not self.calendar_embeddings and not self.cross_calendar_time_context:
            return ref.new_zeros(B, 0)
        out_dim = 0
        if self.calendar_embeddings:
            out_dim += self.calendar_emb_dim * 6
        if self.cross_calendar_time_context:
            out_dim += self.calendar_emb_dim * 3
        idxs = self.build_time_indices(timestamp, ref)
        if idxs is None:
            return ref.new_zeros(B, out_dim)
        hour, hour_cn, weekday, weekday_cn, period3, period6 = idxs
        feats = []
        if self.calendar_embeddings:
            feats.extend([
                self.hour_emb(hour),
                self.hour_cn_emb(hour_cn),
                self.weekday_emb(weekday),
                self.weekday_cn_emb(weekday_cn),
                self.period3_emb(period3),
                self.period6_emb(period6),
            ])
        if self.cross_calendar_time_context:
            weekend_cn = (weekday_cn >= 5).long()
            hour_weekday = (hour_cn * 7 + weekday_cn).clamp(0, 24 * 7 - 1)
            period3_weekday = (period3 * 7 + weekday_cn).clamp(0, 8 * 7 - 1)
            weekend_hour = (weekend_cn * 24 + hour_cn).clamp(0, 2 * 24 - 1)
            feats.extend([
                self.hour_weekday_emb(hour_weekday),
                self.period3_weekday_emb(period3_weekday),
                self.weekend_hour_emb(weekend_hour),
            ])
        return torch.cat(feats, dim=-1)

    def forward(
        self,
        timestamp: Optional[torch.Tensor],
        dense_ctx: torch.Tensor,
        user_ctx: torch.Tensor,
    ) -> torch.Tensor:
        time_feats = self.build_time_features(
            timestamp, dense_ctx, multi_resolution=self.multi_resolution)
        cal_feats = self._calendar_embedding_features(timestamp, dense_ctx)
        return self.weight * self.proj(
            torch.cat([dense_ctx, user_ctx, time_feats, cal_feats], dim=-1))


class UserTimeFiLMModulator(nn.Module):
    """Tiny exposure-time FiLM for user-side tokens only."""

    def __init__(
        self,
        d_model: int,
        weight: float = 0.015,
        multi_resolution: bool = False,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.multi_resolution = bool(multi_resolution)
        time_dim = ExposureTimeContext.feature_dim(self.multi_resolution)
        self.token_norm = nn.LayerNorm(d_model)
        self.modulator = nn.Sequential(
            nn.LayerNorm(time_dim),
            nn.Linear(time_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2),
        )
        nn.init.zeros_(self.modulator[-1].weight)
        nn.init.zeros_(self.modulator[-1].bias)

    def _apply_modulation(
        self,
        tokens: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        delta = torch.tanh(scale).unsqueeze(1) * self.token_norm(tokens) + shift.unsqueeze(1)
        return tokens + self.weight * delta

    def forward(
        self,
        timestamp: Optional[torch.Tensor],
        user_ns: torch.Tensor,
        user_dense_tok: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        ref = user_ns.mean(dim=1)
        time_feats = ExposureTimeContext.build_time_features(
            timestamp, ref, multi_resolution=self.multi_resolution)
        scale, shift = self.modulator(time_feats).chunk(2, dim=-1)
        user_ns = self._apply_modulation(user_ns, scale, shift)
        if user_dense_tok is not None:
            user_dense_tok = self._apply_modulation(user_dense_tok, scale, shift)
        return user_ns, user_dense_tok


class CalendarUserActivityCrossContext(nn.Module):
    """Cross exposure calendar buckets with stable user activity summaries."""

    def __init__(
        self,
        d_model: int,
        num_domains: int,
        weight: float = 0.014,
        calendar_emb_dim: int = 4,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.num_domains = int(num_domains)
        self.calendar_emb_dim = int(calendar_emb_dim)
        self.hour_len_emb = nn.Embedding(24 * 4, self.calendar_emb_dim)
        self.weekday_empty_emb = nn.Embedding(7 * (2 ** self.num_domains), self.calendar_emb_dim)
        self.period3_recent_emb = nn.Embedding(8 * 4, self.calendar_emb_dim)
        self.weekend_dense_emb = nn.Embedding(2 * 4, self.calendar_emb_dim)
        for emb in [
            self.hour_len_emb,
            self.weekday_empty_emb,
            self.period3_recent_emb,
            self.weekend_dense_emb,
        ]:
            nn.init.xavier_normal_(emb.weight.data)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model * 2 + self.calendar_emb_dim * 4),
            nn.Linear(d_model * 2 + self.calendar_emb_dim * 4, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    @staticmethod
    def _bucketize(x: torch.Tensor, edges: List[float]) -> torch.Tensor:
        bucket = torch.zeros_like(x, dtype=torch.long)
        for idx, edge in enumerate(edges, start=1):
            bucket = bucket + (x > edge).long()
        return bucket.clamp(0, len(edges))

    def _activity_indices(
        self,
        timestamp: Optional[torch.Tensor],
        dense_ctx: torch.Tensor,
        seq_time_buckets: Optional[Dict[str, torch.Tensor]],
        seq_masks_list: List[torch.Tensor],
        domains: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = dense_ctx.shape[0]
        device = dense_ctx.device
        if timestamp is None:
            hour_cn = torch.zeros(B, dtype=torch.long, device=device)
            weekday_cn = torch.zeros(B, dtype=torch.long, device=device)
            period3 = torch.zeros(B, dtype=torch.long, device=device)
            weekend_cn = torch.zeros(B, dtype=torch.long, device=device)
        else:
            ts = timestamp.to(device=device, dtype=torch.float32) + 8.0 * 3600.0
            hour_cn = torch.remainder(torch.floor(ts / 3600.0), 24.0).long()
            weekday_cn = torch.remainder(torch.floor(ts / 86400.0) + 4.0, 7.0).long()
            period3 = torch.floor(hour_cn.float() / 3.0).long().clamp(0, 7)
            weekend_cn = (weekday_cn >= 5).long()

        valid_ratios = []
        recent_densities = []
        empty_pattern = torch.zeros(B, dtype=torch.long, device=device)
        for idx, (domain, mask) in enumerate(zip(domains, seq_masks_list)):
            valid = (~mask).float()
            tb = seq_time_buckets.get(domain) if seq_time_buckets is not None else None
            if tb is not None:
                valid = valid * (tb > 0).float()
            valid_ratio = valid.mean(dim=1)
            valid_ratios.append(valid_ratio)
            recent_k = min(64, valid.shape[1])
            recent_densities.append(valid[:, -recent_k:].mean(dim=1))
            empty_pattern = empty_pattern + (valid.sum(dim=1) <= 1.0e-6).long() * (2 ** idx)

        if valid_ratios:
            total_activity = torch.stack(valid_ratios, dim=1).mean(dim=1)
            recent_density = torch.stack(recent_densities, dim=1).mean(dim=1)
        else:
            total_activity = dense_ctx.new_zeros(B)
            recent_density = dense_ctx.new_zeros(B)
        dense_norm = dense_ctx.float().norm(dim=-1) / math.sqrt(float(max(dense_ctx.shape[-1], 1)))

        total_seq_len_bucket = self._bucketize(total_activity, [0.02, 0.12, 0.35])
        recent_density_bucket = self._bucketize(recent_density, [0.02, 0.12, 0.35])
        dense_pair_norm_bucket = self._bucketize(dense_norm, [0.50, 1.00, 1.50])
        empty_pattern = empty_pattern.clamp(0, 2 ** self.num_domains - 1)

        hour_len = (hour_cn * 4 + total_seq_len_bucket).clamp(0, 24 * 4 - 1)
        weekday_empty = (weekday_cn * (2 ** self.num_domains) + empty_pattern).clamp(
            0, 7 * (2 ** self.num_domains) - 1)
        period_recent = (period3 * 4 + recent_density_bucket).clamp(0, 8 * 4 - 1)
        weekend_dense = (weekend_cn * 4 + dense_pair_norm_bucket).clamp(0, 2 * 4 - 1)
        return hour_len, weekday_empty, period_recent, weekend_dense

    def forward(
        self,
        timestamp: Optional[torch.Tensor],
        dense_ctx: torch.Tensor,
        user_ctx: torch.Tensor,
        seq_time_buckets: Optional[Dict[str, torch.Tensor]],
        seq_masks_list: List[torch.Tensor],
        domains: List[str],
    ) -> torch.Tensor:
        hour_len, weekday_empty, period_recent, weekend_dense = self._activity_indices(
            timestamp, dense_ctx, seq_time_buckets, seq_masks_list, domains)
        feats = torch.cat([
            self.hour_len_emb(hour_len),
            self.weekday_empty_emb(weekday_empty),
            self.period3_recent_emb(period_recent),
            self.weekend_dense_emb(weekend_dense),
        ], dim=-1)
        return self.weight * self.proj(torch.cat([dense_ctx, user_ctx, feats], dim=-1))


class HeadRecentActivityCrossContext(nn.Module):
    """Head-window user activity crossed with Beijing-local calendar buckets."""

    def __init__(
        self,
        d_model: int,
        num_domains: int,
        weight: float = 0.006,
        calendar_emb_dim: int = 4,
        head_k: int = 64,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.num_domains = int(num_domains)
        self.calendar_emb_dim = int(calendar_emb_dim)
        self.head_k = int(head_k)
        self.period_domain_valid_emb = nn.Embedding(8 * self.num_domains * 4, self.calendar_emb_dim)
        self.period_domain_short_emb = nn.Embedding(8 * self.num_domains * 4, self.calendar_emb_dim)
        self.weekend_domain_day_emb = nn.Embedding(2 * self.num_domains * 4, self.calendar_emb_dim)
        self.weekday_empty_emb = nn.Embedding(7 * (2 ** self.num_domains), self.calendar_emb_dim)
        for emb in [
            self.period_domain_valid_emb,
            self.period_domain_short_emb,
            self.weekend_domain_day_emb,
            self.weekday_empty_emb,
        ]:
            nn.init.xavier_normal_(emb.weight.data)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model * 2 + self.calendar_emb_dim * 4),
            nn.Linear(d_model * 2 + self.calendar_emb_dim * 4, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    @staticmethod
    def _bucketize(x: torch.Tensor, edges: List[float]) -> torch.Tensor:
        bucket = torch.zeros_like(x, dtype=torch.long)
        for edge in edges:
            bucket = bucket + (x > edge).long()
        return bucket.clamp(0, len(edges))

    def _head_indices(
        self,
        timestamp: Optional[torch.Tensor],
        dense_ctx: torch.Tensor,
        seq_time_buckets: Optional[Dict[str, torch.Tensor]],
        seq_masks_list: List[torch.Tensor],
        domains: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = dense_ctx.shape[0]
        device = dense_ctx.device
        if timestamp is None:
            weekday_cn = torch.zeros(B, dtype=torch.long, device=device)
            period3 = torch.zeros(B, dtype=torch.long, device=device)
            weekend_cn = torch.zeros(B, dtype=torch.long, device=device)
        else:
            ts = timestamp.to(device=device, dtype=torch.float32) + 8.0 * 3600.0
            hour_cn = torch.remainder(torch.floor(ts / 3600.0), 24.0).long()
            weekday_cn = torch.remainder(torch.floor(ts / 86400.0) + 4.0, 7.0).long()
            period3 = torch.floor(hour_cn.float() / 3.0).long().clamp(0, 7)
            weekend_cn = (weekday_cn >= 5).long()

        valid_buckets = []
        short_buckets = []
        day_buckets = []
        empty_pattern = torch.zeros(B, dtype=torch.long, device=device)
        for idx, (domain, mask) in enumerate(zip(domains, seq_masks_list)):
            tb = seq_time_buckets.get(domain) if seq_time_buckets is not None else None
            valid = (~mask).float()
            if tb is not None:
                valid = valid * (tb > 0).float()
            head_k = min(self.head_k, int(valid.shape[1]))
            if head_k <= 0:
                head_valid = valid
                head_tb = tb
            else:
                head_valid = valid[:, :head_k]
                head_tb = tb[:, :head_k] if tb is not None else None
            denom = head_valid.sum(dim=1).clamp(min=1.0)
            valid_ratio = head_valid.mean(dim=1)
            if head_tb is None:
                short_ratio = dense_ctx.new_zeros(B)
                day_ratio = dense_ctx.new_zeros(B)
            else:
                short_ratio = (((head_tb > 0) & (head_tb <= 21)).float() * head_valid).sum(dim=1) / denom
                day_ratio = (((head_tb > 0) & (head_tb <= 47)).float() * head_valid).sum(dim=1) / denom
            valid_buckets.append((idx, self._bucketize(valid_ratio, [0.02, 0.12, 0.35])))
            short_buckets.append((idx, self._bucketize(short_ratio, [0.02, 0.12, 0.35])))
            day_buckets.append((idx, self._bucketize(day_ratio, [0.02, 0.12, 0.35])))
            empty_pattern = empty_pattern + (valid.sum(dim=1) <= 1.0e-6).long() * (2 ** idx)

        if valid_buckets:
            valid_idx = torch.stack([
                (period3 * self.num_domains * 4 + idx * 4 + bucket).clamp(
                    0, 8 * self.num_domains * 4 - 1)
                for idx, bucket in valid_buckets
            ], dim=1)
            short_idx = torch.stack([
                (period3 * self.num_domains * 4 + idx * 4 + bucket).clamp(
                    0, 8 * self.num_domains * 4 - 1)
                for idx, bucket in short_buckets
            ], dim=1)
            day_idx = torch.stack([
                (weekend_cn * self.num_domains * 4 + idx * 4 + bucket).clamp(
                    0, 2 * self.num_domains * 4 - 1)
                for idx, bucket in day_buckets
            ], dim=1)
        else:
            valid_idx = torch.zeros(B, 1, dtype=torch.long, device=device)
            short_idx = torch.zeros(B, 1, dtype=torch.long, device=device)
            day_idx = torch.zeros(B, 1, dtype=torch.long, device=device)
        empty_idx = (weekday_cn * (2 ** self.num_domains) + empty_pattern.clamp(
            0, 2 ** self.num_domains - 1)).clamp(0, 7 * (2 ** self.num_domains) - 1)
        return valid_idx, short_idx, day_idx, empty_idx

    def forward(
        self,
        timestamp: Optional[torch.Tensor],
        dense_ctx: torch.Tensor,
        user_ctx: torch.Tensor,
        seq_time_buckets: Optional[Dict[str, torch.Tensor]],
        seq_masks_list: List[torch.Tensor],
        domains: List[str],
    ) -> torch.Tensor:
        valid_idx, short_idx, day_idx, empty_idx = self._head_indices(
            timestamp, dense_ctx, seq_time_buckets, seq_masks_list, domains)
        valid_emb = self.period_domain_valid_emb(valid_idx).mean(dim=1)
        short_emb = self.period_domain_short_emb(short_idx).mean(dim=1)
        day_emb = self.weekend_domain_day_emb(day_idx).mean(dim=1)
        feats = torch.cat([
            valid_emb,
            short_emb,
            day_emb,
            self.weekday_empty_emb(empty_idx),
        ], dim=-1)
        return self.weight * self.proj(torch.cat([dense_ctx, user_ctx, feats], dim=-1))


class UserFieldCoverageTimeContext(nn.Module):
    """Low-card user field coverage profile crossed with Beijing-local time."""

    DEFAULT_USER_GROUP_FIDS = (
        (1, 15),
        (48, 49, 89, 90, 91),
        (80,),
        (51, 52, 53, 54, 86),
        (82, 92, 93),
        (50, 60, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109),
        (3, 4, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66),
    )
    PAIR_FIDS = (62, 63, 64, 65, 66, 89, 90, 91)

    def __init__(
        self,
        d_model: int,
        user_int_feature_specs: List[Tuple[int, int, int, int]],
        num_domains: int,
        weight: float = 0.006,
        calendar_emb_dim: int = 4,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.num_domains = int(num_domains)
        self.calendar_emb_dim = int(calendar_emb_dim)
        self.group_slices: List[List[Tuple[int, int]]] = []
        fid_to_slice = {int(fid): (int(offset), int(length))
                        for fid, _vs, offset, length in user_int_feature_specs}
        for group in self.DEFAULT_USER_GROUP_FIDS:
            slices = [fid_to_slice[fid] for fid in group if fid in fid_to_slice]
            self.group_slices.append(slices)
        self.pair_slices = [fid_to_slice[fid] for fid in self.PAIR_FIDS if fid in fid_to_slice]
        self.num_groups = len(self.group_slices)

        self.period_group_cov_emb = nn.Embedding(8 * self.num_groups * 4, self.calendar_emb_dim)
        self.period_group_multi_emb = nn.Embedding(8 * self.num_groups * 4, self.calendar_emb_dim)
        self.weekend_pair_emb = nn.Embedding(2 * 4, self.calendar_emb_dim)
        self.weekday_empty_emb = nn.Embedding(7 * (2 ** self.num_domains), self.calendar_emb_dim)
        self.dense_norm_emb = nn.Embedding(4, self.calendar_emb_dim)
        for emb in [
            self.period_group_cov_emb,
            self.period_group_multi_emb,
            self.weekend_pair_emb,
            self.weekday_empty_emb,
            self.dense_norm_emb,
        ]:
            nn.init.xavier_normal_(emb.weight.data)
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model * 2 + self.calendar_emb_dim * 5),
            nn.Linear(d_model * 2 + self.calendar_emb_dim * 5, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    @staticmethod
    def _bucketize(x: torch.Tensor, edges: List[float]) -> torch.Tensor:
        bucket = torch.zeros_like(x, dtype=torch.long)
        for edge in edges:
            bucket = bucket + (x > edge).long()
        return bucket.clamp(0, len(edges))

    @staticmethod
    def _segment_presence(user_int_feats: torch.Tensor, slices: List[Tuple[int, int]]) -> torch.Tensor:
        if not slices:
            return user_int_feats.new_zeros(user_int_feats.shape[0])
        parts = []
        for offset, length in slices:
            segment = user_int_feats[:, offset:offset + length]
            parts.append((segment > 0).float().mean(dim=1))
        return torch.stack(parts, dim=1).mean(dim=1)

    def _empty_pattern(
        self,
        seq_time_buckets: Optional[Dict[str, torch.Tensor]],
        seq_masks_list: List[torch.Tensor],
        domains: List[str],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        B = ref.shape[0]
        empty_pattern = torch.zeros(B, dtype=torch.long, device=ref.device)
        for idx, (domain, mask) in enumerate(zip(domains, seq_masks_list)):
            valid = (~mask).float()
            tb = seq_time_buckets.get(domain) if seq_time_buckets is not None else None
            if tb is not None:
                valid = valid * (tb > 0).float()
            empty_pattern = empty_pattern + (valid.sum(dim=1) <= 1.0e-6).long() * (2 ** idx)
        return empty_pattern.clamp(0, 2 ** self.num_domains - 1)

    def forward(
        self,
        timestamp: Optional[torch.Tensor],
        dense_ctx: torch.Tensor,
        user_ctx: torch.Tensor,
        user_int_feats: torch.Tensor,
        seq_time_buckets: Optional[Dict[str, torch.Tensor]],
        seq_masks_list: List[torch.Tensor],
        domains: List[str],
    ) -> torch.Tensor:
        B = dense_ctx.shape[0]
        device = dense_ctx.device
        if timestamp is None:
            weekday_cn = torch.zeros(B, dtype=torch.long, device=device)
            period3 = torch.zeros(B, dtype=torch.long, device=device)
            weekend_cn = torch.zeros(B, dtype=torch.long, device=device)
        else:
            ts = timestamp.to(device=device, dtype=torch.float32) + 8.0 * 3600.0
            hour_cn = torch.remainder(torch.floor(ts / 3600.0), 24.0).long()
            weekday_cn = torch.remainder(torch.floor(ts / 86400.0) + 4.0, 7.0).long()
            period3 = torch.floor(hour_cn.float() / 3.0).long().clamp(0, 7)
            weekend_cn = (weekday_cn >= 5).long()

        group_cov = []
        group_multi = []
        for slices in self.group_slices:
            presence = self._segment_presence(user_int_feats, slices)
            group_cov.append(self._bucketize(presence, [0.02, 0.20, 0.60]))
            group_multi.append(self._bucketize(presence, [0.05, 0.35, 0.75]))
        if group_cov:
            cov_idx = torch.stack([
                (period3 * self.num_groups * 4 + idx * 4 + bucket).clamp(
                    0, 8 * self.num_groups * 4 - 1)
                for idx, bucket in enumerate(group_cov)
            ], dim=1)
            multi_idx = torch.stack([
                (period3 * self.num_groups * 4 + idx * 4 + bucket).clamp(
                    0, 8 * self.num_groups * 4 - 1)
                for idx, bucket in enumerate(group_multi)
            ], dim=1)
        else:
            cov_idx = torch.zeros(B, 1, dtype=torch.long, device=device)
            multi_idx = torch.zeros(B, 1, dtype=torch.long, device=device)

        pair_presence = self._segment_presence(user_int_feats, self.pair_slices)
        pair_bucket = self._bucketize(pair_presence, [0.02, 0.30, 0.70])
        pair_idx = (weekend_cn * 4 + pair_bucket).clamp(0, 2 * 4 - 1)
        dense_norm = dense_ctx.float().norm(dim=-1) / math.sqrt(float(max(dense_ctx.shape[-1], 1)))
        dense_bucket = self._bucketize(dense_norm, [0.50, 1.00, 1.50])
        empty_idx = (
            weekday_cn * (2 ** self.num_domains)
            + self._empty_pattern(seq_time_buckets, seq_masks_list, domains, dense_ctx)
        ).clamp(0, 7 * (2 ** self.num_domains) - 1)

        feats = torch.cat([
            self.period_group_cov_emb(cov_idx).mean(dim=1),
            self.period_group_multi_emb(multi_idx).mean(dim=1),
            self.weekend_pair_emb(pair_idx),
            self.weekday_empty_emb(empty_idx),
            self.dense_norm_emb(dense_bucket),
        ], dim=-1)
        return self.weight * self.proj(torch.cat([dense_ctx, user_ctx, feats], dim=-1))


class DenseSemanticTimeBilinearContext(nn.Module):
    """Low-rank interaction between DensePair components and calendar time."""

    def __init__(
        self,
        d_model: int,
        rank: int = 8,
        weight: float = 0.012,
        calendar_emb_dim: int = 4,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.weight = float(weight)
        self.calendar_emb_dim = int(calendar_emb_dim)
        self.time_dim = ExposureTimeContext.feature_dim(True) + self.calendar_emb_dim * 9
        self.hour_emb = nn.Embedding(24, self.calendar_emb_dim)
        self.hour_cn_emb = nn.Embedding(24, self.calendar_emb_dim)
        self.weekday_emb = nn.Embedding(7, self.calendar_emb_dim)
        self.weekday_cn_emb = nn.Embedding(7, self.calendar_emb_dim)
        self.period3_emb = nn.Embedding(8, self.calendar_emb_dim)
        self.period6_emb = nn.Embedding(4, self.calendar_emb_dim)
        self.hour_weekday_emb = nn.Embedding(24 * 7, self.calendar_emb_dim)
        self.period3_weekday_emb = nn.Embedding(8 * 7, self.calendar_emb_dim)
        self.weekend_hour_emb = nn.Embedding(2 * 24, self.calendar_emb_dim)
        for emb in [
            self.hour_emb, self.hour_cn_emb, self.weekday_emb, self.weekday_cn_emb,
            self.period3_emb, self.period6_emb, self.hour_weekday_emb,
            self.period3_weekday_emb, self.weekend_hour_emb,
        ]:
            nn.init.xavier_normal_(emb.weight.data)
        self.time_down = nn.Sequential(
            nn.LayerNorm(self.time_dim),
            nn.Linear(self.time_dim, self.rank),
        )
        self.semantic_down = nn.Linear(d_model, self.rank, bias=False)
        self.scalar_down = nn.Linear(d_model, self.rank, bias=False)
        self.pair_down = nn.Linear(d_model, self.rank, bias=False)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model * 3 + self.rank * 3),
            nn.Linear(d_model * 3 + self.rank * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def _calendar_feats(
        self,
        timestamp: Optional[torch.Tensor],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        B = ref.shape[0]
        if timestamp is None:
            return ref.new_zeros(B, self.time_dim)
        base = ExposureTimeContext.build_time_features(
            timestamp, ref, multi_resolution=True)
        idxs = ExposureTimeContext.build_time_indices(timestamp, ref)
        if idxs is None:
            return ref.new_zeros(B, self.time_dim)
        hour, hour_cn, weekday, weekday_cn, period3, period6 = idxs
        weekend_cn = (weekday_cn >= 5).long()
        hour_weekday = (hour_cn * 7 + weekday_cn).clamp(0, 24 * 7 - 1)
        period3_weekday = (period3 * 7 + weekday_cn).clamp(0, 8 * 7 - 1)
        weekend_hour = (weekend_cn * 24 + hour_cn).clamp(0, 2 * 24 - 1)
        cal = torch.cat([
            self.hour_emb(hour),
            self.hour_cn_emb(hour_cn),
            self.weekday_emb(weekday),
            self.weekday_cn_emb(weekday_cn),
            self.period3_emb(period3),
            self.period6_emb(period6),
            self.hour_weekday_emb(hour_weekday),
            self.period3_weekday_emb(period3_weekday),
            self.weekend_hour_emb(weekend_hour),
        ], dim=-1)
        return torch.cat([base, cal], dim=-1)

    def forward(
        self,
        timestamp: Optional[torch.Tensor],
        semantic_ctx: torch.Tensor,
        scalar_ctx: torch.Tensor,
        pair_ctx: torch.Tensor,
    ) -> torch.Tensor:
        time_low = self.time_down(self._calendar_feats(timestamp, pair_ctx))
        sem_cross = self.semantic_down(semantic_ctx) * time_low
        scalar_cross = self.scalar_down(scalar_ctx) * time_low
        pair_cross = self.pair_down(pair_ctx) * time_low
        return self.weight * self.out(torch.cat([
            semantic_ctx, scalar_ctx, pair_ctx, sem_cross, scalar_cross, pair_cross
        ], dim=-1))


class TimeDeltaHistogramSidecar(nn.Module):
    """Per-domain sequence recency histogram used only as query evidence."""

    def __init__(
        self,
        d_model: int,
        num_domains: int,
        weight: float = 0.015,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.num_domains = int(num_domains)
        self.edges = (21, 31, 47, 64)
        self.feat_dim = self.num_domains * (len(self.edges) + 2)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.feat_dim),
            nn.Linear(self.feat_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(
        self,
        seq_time_buckets: Dict[str, torch.Tensor],
        seq_masks_list: List[torch.Tensor],
        domains: List[str],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        feats = []
        B = ref.shape[0]
        last_edge = 0
        for domain, mask in zip(domains, seq_masks_list):
            tb = seq_time_buckets.get(domain)
            if tb is None:
                feats.append(ref.new_zeros(B, len(self.edges) + 2))
                continue
            valid = ((~mask) & (tb > 0)).float()
            valid_sum = valid.sum(dim=1, keepdim=True)
            denom = valid_sum.clamp(min=1.0)
            domain_feats = []
            prev = 0
            for edge in self.edges:
                in_window = ((tb > prev) & (tb <= edge)).float() * valid
                domain_feats.append(in_window.sum(dim=1, keepdim=True) / denom)
                prev = edge
            valid_len = valid.mean(dim=1, keepdim=True)
            empty = (valid_sum <= 1.0e-6).float()
            domain_feats.extend([valid_len, empty])
            feats.append(torch.cat(domain_feats, dim=-1))
            last_edge = prev
        _ = last_edge  # keep the window definition explicit for logs/debugging.
        return self.weight * self.proj(torch.cat(feats, dim=-1))


class UserTimeEvidenceRecBlock(nn.Module):
    """Tiny MetaFormer-style mixer over user, dense-pair, exposure-time, and time-delta evidence."""

    def __init__(
        self,
        d_model: int,
        weight: float = 0.015,
        multi_resolution: bool = True,
        calendar_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.multi_resolution = bool(multi_resolution)
        self.calendar_embeddings = bool(calendar_embeddings)
        time_dim = ExposureTimeContext.feature_dim(self.multi_resolution)
        emb_dim = 24 if self.calendar_embeddings else 0
        self.time_proj = nn.Sequential(
            nn.LayerNorm(time_dim + emb_dim),
            nn.Linear(time_dim + emb_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        if self.calendar_embeddings:
            self.time_calendar = ExposureTimeContext(
                d_model=d_model,
                weight=1.0,
                multi_resolution=self.multi_resolution,
                calendar_embeddings=True,
                calendar_emb_dim=4,
            )
        self.token_norm = nn.LayerNorm(d_model)
        self.token_mixer = nn.Sequential(
            nn.Linear(4, 8),
            nn.SiLU(),
            nn.Linear(8, 4),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(
        self,
        timestamp: Optional[torch.Tensor],
        dense_ctx: torch.Tensor,
        user_ctx: torch.Tensor,
        sidecar_ctx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        time_feats = ExposureTimeContext.build_time_features(
            timestamp, dense_ctx, multi_resolution=self.multi_resolution)
        if self.calendar_embeddings:
            cal_feats = self.time_calendar._calendar_embedding_features(timestamp, dense_ctx)
            time_feats = torch.cat([time_feats, cal_feats], dim=-1)
        time_ctx = self.time_proj(time_feats)
        if sidecar_ctx is None:
            sidecar_ctx = dense_ctx.new_zeros(dense_ctx.shape)
        x = torch.stack([dense_ctx, user_ctx, time_ctx, sidecar_ctx], dim=1)
        mixed = self.token_mixer(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + mixed
        x = x + self.ffn(x)
        return self.weight * self.out(x.mean(dim=1))


class TargetLiteDomainRouter(nn.Module):
    """Low-card target and user-time evidence context for safer domain routing."""

    def __init__(
        self,
        d_model: int,
        emb_dim: int,
        item_int_feature_specs: List[Tuple[int, ...]],
        item_dense_dim: int,
        num_domains: int,
        low_card_threshold: int = 1000,
        weight: float = 0.012,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.item_int_feature_specs = item_int_feature_specs
        self.low_card_indices = [
            i for i, spec in enumerate(item_int_feature_specs)
            if 0 < _unpack_feature_spec(spec)[1] <= int(low_card_threshold)
        ]
        self.low_card_embs = nn.ModuleList()
        for idx in self.low_card_indices:
            _, vs, _, _ = _unpack_feature_spec(item_int_feature_specs[idx])
            emb = nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0)
            nn.init.xavier_normal_(emb.weight.data)
            emb.weight.data[0, :] = 0
            self.low_card_embs.append(emb)
        self.target_low_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.LayerNorm(item_dense_dim),
                nn.Linear(item_dense_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
        self.domain_feat_dim = num_domains * 3
        self.out = nn.Sequential(
            nn.LayerNorm(d_model * 4 + self.domain_feat_dim),
            nn.Linear(d_model * 4 + self.domain_feat_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def _target_low_card_context(self, item_int_feats: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if not self.low_card_indices:
            return ref.new_zeros(ref.shape)
        field_embs = []
        for emb, idx in zip(self.low_card_embs, self.low_card_indices):
            _, vs, offset, length = _unpack_feature_spec(self.item_int_feature_specs[idx])
            vals = item_int_feats[:, offset: offset + length].clamp(min=0, max=int(vs)).long()
            e = emb(vals)
            mask = (vals > 0).float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            field_embs.append((e * mask).sum(dim=1) / denom)
        low = torch.stack(field_embs, dim=1).mean(dim=1)
        return self.target_low_proj(low)

    def _domain_features(
        self,
        seq_time_buckets: Optional[Dict[str, torch.Tensor]],
        seq_masks_list: List[torch.Tensor],
        domains: List[str],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        feats = []
        B = ref.shape[0]
        for domain, mask in zip(domains, seq_masks_list):
            tb = seq_time_buckets.get(domain) if seq_time_buckets is not None else None
            if tb is None:
                feats.append(ref.new_zeros(B, 3))
                continue
            valid = ((~mask) & (tb > 0)).float()
            valid_sum = valid.sum(dim=1, keepdim=True)
            denom = valid_sum.clamp(min=1.0)
            valid_len = valid.mean(dim=1, keepdim=True)
            empty = (valid_sum <= 1.0e-6).float()
            recent = (((tb > 0) & (tb <= 21)).float() * valid).sum(dim=1, keepdim=True) / denom
            feats.append(torch.cat([valid_len, empty, recent], dim=-1))
        return torch.cat(feats, dim=-1)

    def forward(
        self,
        item_int_feats: torch.Tensor,
        item_dense_feats: Optional[torch.Tensor],
        dense_ctx: torch.Tensor,
        user_ctx: torch.Tensor,
        seq_time_buckets: Optional[Dict[str, torch.Tensor]],
        seq_masks_list: List[torch.Tensor],
        domains: List[str],
    ) -> torch.Tensor:
        target_low = self._target_low_card_context(item_int_feats, dense_ctx)
        if self.has_item_dense and item_dense_feats is not None:
            target_dense = self.item_dense_proj(item_dense_feats)
        else:
            target_dense = dense_ctx.new_zeros(dense_ctx.shape)
        domain_feats = self._domain_features(seq_time_buckets, seq_masks_list, domains, dense_ctx)
        return self.weight * self.out(torch.cat([
            dense_ctx, user_ctx, target_low, target_dense, domain_feats
        ], dim=-1))


class LowCardTemporalContentSidecar(nn.Module):
    """Evidence block input from low-card sequence content across time windows."""

    def __init__(
        self,
        d_model: int,
        num_domains: int,
        weight: float = 0.012,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.num_domains = int(num_domains)
        self.num_windows = 4
        token_count = self.num_domains * self.num_windows
        self.token_norm = nn.LayerNorm(d_model)
        self.token_mixer = nn.Sequential(
            nn.Linear(token_count, token_count),
            nn.SiLU(),
            nn.Linear(token_count, token_count),
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, content_tokens: torch.Tensor) -> torch.Tensor:
        x = content_tokens
        mixed = self.token_mixer(self.token_norm(x).transpose(1, 2)).transpose(1, 2)
        x = x + mixed
        x = x + self.ffn(x)
        return self.weight * self.out(x.mean(dim=1))


class UserTimeItemEvidenceMixer(nn.Module):
    """Low-rank CrossNet over user, dense-pair, item and exposure-time evidence."""

    def __init__(self, d_model: int, rank: int = 8, weight: float = 0.015) -> None:
        super().__init__()
        self.weight = float(weight)
        self.input_dim = d_model * 3 + 7
        self.norm = nn.LayerNorm(self.input_dim)
        self.downs = nn.ModuleList([
            nn.Linear(self.input_dim, rank, bias=False)
            for _ in range(2)
        ])
        self.ups = nn.ModuleList([
            nn.Linear(rank, self.input_dim, bias=True)
            for _ in range(2)
        ])
        self.out = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(
        self,
        timestamp: Optional[torch.Tensor],
        dense_ctx: torch.Tensor,
        user_ctx: torch.Tensor,
        item_ctx: torch.Tensor,
    ) -> torch.Tensor:
        time_feats = ExposureTimeContext.build_time_features(timestamp, dense_ctx)
        x0 = self.norm(torch.cat([dense_ctx, user_ctx, item_ctx, time_feats], dim=-1))
        x = x0
        for down, up in zip(self.downs, self.ups):
            crossed = up(F.gelu(down(x)))
            x = self.norm(x + x0 * crossed)
        return self.weight * self.out(x)


class UserTimeSegmentExpertContext(nn.Module):
    """Small user-time expert context without touching existing NS/query tokens."""

    def __init__(
        self,
        d_model: int,
        weight: float = 0.018,
        multi_resolution: bool = True,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.multi_resolution = bool(multi_resolution)
        time_dim = ExposureTimeContext.feature_dim(self.multi_resolution)
        self.input_dim = d_model * 2 + time_dim
        self.router = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 4),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.input_dim),
                nn.Linear(self.input_dim, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(4)
        ])
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        for expert in self.experts:
            nn.init.zeros_(expert[-1].weight)
            nn.init.zeros_(expert[-1].bias)

    def forward(
        self,
        timestamp: Optional[torch.Tensor],
        dense_ctx: torch.Tensor,
        user_ctx: torch.Tensor,
    ) -> torch.Tensor:
        time_feats = ExposureTimeContext.build_time_features(
            timestamp, dense_ctx, multi_resolution=self.multi_resolution)
        x = torch.cat([dense_ctx, user_ctx, time_feats], dim=-1)
        router_logits = self.router(x)
        idxs = ExposureTimeContext.build_time_indices(timestamp, dense_ctx)
        if idxs is not None:
            _, hour_cn, _, weekday_cn, _, _ = idxs
            is_weekend = weekday_cn >= 5
            is_daytime = (hour_cn >= 7) & (hour_cn < 19)
            segment = torch.zeros_like(hour_cn)
            segment = torch.where((~is_weekend) & (~is_daytime), segment.new_full(segment.shape, 1), segment)
            segment = torch.where(is_weekend & is_daytime, segment.new_full(segment.shape, 2), segment)
            segment = torch.where(is_weekend & (~is_daytime), segment.new_full(segment.shape, 3), segment)
            segment_prior = F.one_hot(segment.clamp(0, 3), num_classes=4).to(router_logits.dtype)
            router_logits = router_logits + segment_prior
        router_weight = torch.softmax(router_logits, dim=-1)
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed = (expert_out * router_weight.unsqueeze(-1)).sum(dim=1)
        return self.weight * mixed


class CalendarDomainRouter(nn.Module):
    """Calendar/user evidence router that adds tiny per-domain query deltas."""

    def __init__(
        self,
        d_model: int,
        num_domains: int,
        weight: float = 0.010,
        multi_resolution: bool = True,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.num_domains = int(num_domains)
        self.multi_resolution = bool(multi_resolution)
        self.time_dim = ExposureTimeContext.feature_dim(self.multi_resolution)
        self.domain_feat_dim = self.num_domains * 3
        input_dim = d_model * 2 + self.time_dim + self.domain_feat_dim
        self.router = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, self.num_domains),
        )
        self.delta_proj = nn.Sequential(
            nn.LayerNorm(input_dim + self.num_domains),
            nn.Linear(input_dim + self.num_domains, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, self.num_domains * d_model),
        )
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        nn.init.zeros_(self.delta_proj[-1].weight)
        nn.init.zeros_(self.delta_proj[-1].bias)

    def _domain_features(
        self,
        seq_time_buckets: Optional[Dict[str, torch.Tensor]],
        seq_masks_list: List[torch.Tensor],
        domains: List[str],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        feats = []
        B = ref.shape[0]
        for domain, mask in zip(domains, seq_masks_list):
            tb = seq_time_buckets.get(domain) if seq_time_buckets is not None else None
            if tb is None:
                feats.append(ref.new_zeros(B, 3))
                continue
            valid = ((~mask) & (tb > 0)).float()
            valid_sum = valid.sum(dim=1, keepdim=True)
            denom = valid_sum.clamp(min=1.0)
            valid_len = valid.mean(dim=1, keepdim=True)
            empty = (valid_sum <= 1.0e-6).float()
            recent_density = (((tb > 0) & (tb <= 21)).float() * valid).sum(
                dim=1, keepdim=True) / denom
            feats.append(torch.cat([valid_len, empty, recent_density], dim=-1))
        return torch.cat(feats, dim=-1)

    def forward(
        self,
        timestamp: Optional[torch.Tensor],
        dense_ctx: torch.Tensor,
        user_ctx: torch.Tensor,
        seq_time_buckets: Optional[Dict[str, torch.Tensor]],
        seq_masks_list: List[torch.Tensor],
        domains: List[str],
    ) -> torch.Tensor:
        time_feats = ExposureTimeContext.build_time_features(
            timestamp, dense_ctx, multi_resolution=self.multi_resolution)
        domain_feats = self._domain_features(seq_time_buckets, seq_masks_list, domains, dense_ctx)
        x = torch.cat([dense_ctx, user_ctx, time_feats, domain_feats], dim=-1)
        domain_weight = torch.softmax(self.router(x), dim=-1)
        delta = self.delta_proj(torch.cat([x, domain_weight], dim=-1))
        delta = delta.view(dense_ctx.shape[0], self.num_domains, -1)
        return self.weight * domain_weight.unsqueeze(-1) * delta


class MinimalHRRMDQAdapter(nn.Module):
    """Domain-quality query adapter with recent/long/reverse memories."""

    def __init__(
        self,
        d_model: int,
        hidden_mult: int = 2,
        recent_k: int = 64,
        query_weight: float = 0.02,
        memory_weight: float = 0.03,
    ) -> None:
        super().__init__()
        self.recent_k = max(1, int(recent_k))
        self.query_weight = float(query_weight)
        self.memory_weight = float(memory_weight)
        hidden = d_model * hidden_mult

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        self.score_mlp = nn.Sequential(
            nn.LayerNorm(d_model + 6),
            nn.Linear(d_model + 6, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        self.memory_delta_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        for seq in [self.memory_proj, self.score_mlp, self.query_proj, self.memory_delta_proj]:
            nn.init.zeros_(seq[-1].weight)
            nn.init.zeros_(seq[-1].bias)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weights = valid.float()
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return torch.bmm(weights.unsqueeze(1), tokens).squeeze(1) / denom

    def _weighted_summary(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        B, L, _ = tokens.shape
        device = tokens.device
        valid_f = valid.float()
        pos = torch.arange(L, device=device, dtype=torch.float).unsqueeze(0).expand(B, -1)
        valid_len = valid_f.sum(dim=1)

        if mode == 'recent':
            start = (valid_len - self.recent_k).clamp(min=0).unsqueeze(1)
            weights = ((pos >= start) & (pos < valid_len.unsqueeze(1))).float()
        elif mode == 'reverse':
            weights = (1.0 - pos / float(max(L - 1, 1))) * valid_f
        else:
            weights = valid_f

        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return torch.bmm(weights.unsqueeze(1), tokens).squeeze(1) / denom

    def _domain_stats(
        self,
        target_ctx: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L, _ = tokens.shape
        valid = ~padding_mask
        valid_f = valid.float()
        denom = valid_f.sum(dim=1).clamp(min=1.0)
        empty = (valid_f.sum(dim=1) == 0).float()

        length_norm = denom / float(max(L, 1))
        recent_k = min(self.recent_k, L)
        recent_density = valid_f[:, -recent_k:].mean(dim=1)

        sim = F.cosine_similarity(tokens, target_ctx.unsqueeze(1), dim=-1)
        sim_mean = (sim * valid_f).sum(dim=1) / denom
        sim_max = sim.masked_fill(~valid, -1.0).max(dim=1).values
        sim_max = torch.where(empty.bool(), sim.new_zeros(B), sim_max)
        sim_var = (((sim - sim_mean.unsqueeze(1)) ** 2) * valid_f).sum(dim=1) / denom
        sim_std = torch.sqrt(sim_var + 1.0e-6)

        return torch.stack([
            length_norm,
            empty,
            recent_density,
            sim_mean,
            sim_max,
            sim_std,
        ], dim=-1)

    def forward(
        self,
        q_tokens_list: List[torch.Tensor],
        target_ctx: torch.Tensor,
        seq_tokens_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        memories = []
        scores = []
        for tokens, mask in zip(seq_tokens_list, seq_masks_list):
            valid = ~mask
            long_mem = self._weighted_summary(tokens, valid, 'long')
            recent_mem = self._weighted_summary(tokens, valid, 'recent')
            reverse_mem = self._weighted_summary(tokens, valid, 'reverse')
            memory = self.memory_proj(torch.cat(
                [target_ctx, long_mem, recent_mem, reverse_mem], dim=-1))
            stats = self._domain_stats(target_ctx, tokens, mask)
            score = self.score_mlp(torch.cat([memory, stats], dim=-1)).squeeze(-1)
            memories.append(memory)
            scores.append(score)

        gates = torch.softmax(torch.stack(scores, dim=1), dim=1)
        out = []
        for i, (q_tokens, memory) in enumerate(zip(q_tokens_list, memories)):
            gate = gates[:, i].view(-1, 1, 1)
            query_delta = self.query_proj(torch.cat([target_ctx, memory], dim=-1))
            memory_delta = self.memory_delta_proj(memory)
            delta = self.query_weight * query_delta + self.memory_weight * memory_delta
            out.append(q_tokens + gate * delta.unsqueeze(1))
        return out


class PCVRHyFormer(nn.Module):
    """PCVRHyFormer model for post-click conversion rate prediction.

    Combines MultiSeqHyFormerBlock and MultiSeqQueryGenerator to process
    multiple input sequences with non-sequence features.
    """

    def __init__(
        self,
        # Data schema
        user_int_feature_specs: List[Tuple[int, int, int]],
        item_int_feature_specs: List[Tuple[int, int, int]],
        user_dense_dim: int,
        item_dense_dim: int,
        seq_vocab_sizes: "dict[str, List[int]]",  # {domain: [vocab_size_per_fid, ...]}
        # NS grouping config (grouped by fid index)
        user_ns_groups: List[List[int]],
        item_ns_groups: List[List[int]],
        user_dense_feature_specs: Optional[List[Tuple[int, int, int]]] = None,
        # Model hyperparameters
        d_model: int = 64,
        emb_dim: int = 64,
        num_queries: int = 1,
        num_hyformer_blocks: int = 2,
        num_heads: int = 4,
        seq_encoder_type: str = 'transformer',
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        seq_top_k: int = 50,
        seq_causal: bool = False,
        action_num: int = 1,
        num_time_buckets: int = 65,
        rank_mixer_mode: str = 'full',
        use_rope: bool = False,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        # NS tokenizer variant
        ns_tokenizer_type: str = 'rankmixer',
        user_ns_tokens: int = 0,
        item_ns_tokens: int = 0,
        role_pair_fids: str = '62,63,64,65,66,89,90,91',
        role_low_card_threshold: int = 1000,
        role_high_card_threshold: int = 100000,
        dense_pair_compressor: bool = False,
        dense_pair_fids: str = '62,63,64,65,66,89,90,91',
        dense_pair_high_dim_threshold: int = 128,
        dense_pair_weight: float = 0.10,
        semantic_seq_tokenizer: bool = False,
        seq_low_card_threshold: int = 1000,
        seq_high_card_residual_weight: float = 0.08,
        use_prequery_role_context: bool = False,
        prequery_context_weight: float = 0.05,
        item_crossnet: bool = False,
        item_cross_rank: int = 16,
        item_cross_weight: float = 0.025,
        time_attention_bias: bool = False,
        time_attention_clip: float = 0.10,
        exposure_time_context: bool = False,
        exposure_time_weight: float = 0.02,
        multi_res_exposure_time: bool = False,
        calendar_time_embeddings: bool = False,
        cross_calendar_time_context: bool = False,
        cross_calendar_time_weight: float = 0.028,
        user_time_segment_experts: bool = False,
        segment_expert_weight: float = 0.018,
        calendar_domain_router: bool = False,
        calendar_domain_router_weight: float = 0.010,
        query_ranklift_regularizer: bool = False,
        query_ranklift_weight: float = 1.0e-4,
        query_ranklift_warmup_epoch: int = 2,
        user_time_film: bool = False,
        user_time_film_weight: float = 0.015,
        time_delta_sidecar: bool = False,
        time_delta_sidecar_weight: float = 0.015,
        user_time_evidence_block: bool = False,
        user_time_evidence_weight: float = 0.015,
        target_lite_domain_router: bool = False,
        target_lite_low_card_threshold: int = 1000,
        target_lite_weight: float = 0.012,
        low_card_temporal_content_sidecar: bool = False,
        low_card_content_weight: float = 0.012,
        query_conditioned_time_attention: bool = False,
        query_time_rank: int = 8,
        user_time_item_mixer: bool = False,
        user_time_item_rank: int = 8,
        user_time_item_weight: float = 0.015,
        calendar_user_activity_cross: bool = False,
        calendar_user_activity_weight: float = 0.014,
        head_recent_activity_cross: bool = False,
        head_recent_activity_weight: float = 0.006,
        head_recent_activity_k: int = 64,
        user_field_coverage_time_context: bool = False,
        user_field_coverage_time_weight: float = 0.006,
        dense_semantic_time_bilinear: bool = False,
        dense_semantic_time_rank: int = 8,
        dense_semantic_time_weight: float = 0.012,
        hrrm_dq_adapter: bool = False,
        hrrm_query_weight: float = 0.02,
        hrrm_memory_weight: float = 0.03,
        hrrm_recent_k: int = 64,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_queries = num_queries
        self.seq_domains = sorted(seq_vocab_sizes.keys())  # deterministic order
        self.num_sequences = len(self.seq_domains)
        self.num_time_buckets = num_time_buckets
        self.rank_mixer_mode = rank_mixer_mode
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.ns_tokenizer_type = ns_tokenizer_type
        self.semantic_seq_tokenizer = bool(semantic_seq_tokenizer)
        self.seq_low_card_threshold = int(seq_low_card_threshold)
        self.seq_high_card_residual_weight = float(seq_high_card_residual_weight)
        self.use_prequery_role_context = bool(use_prequery_role_context)
        self.prequery_context_weight = float(prequery_context_weight)
        self.item_crossnet = bool(item_crossnet)
        self.item_cross_weight = float(item_cross_weight)
        self.time_attention_bias = bool(time_attention_bias)
        self.time_attention_clip = float(time_attention_clip)
        self.exposure_time_context = bool(exposure_time_context)
        self.exposure_time_weight = float(exposure_time_weight)
        self.multi_res_exposure_time = bool(multi_res_exposure_time)
        self.calendar_time_embeddings = bool(calendar_time_embeddings)
        self.cross_calendar_time_context = bool(cross_calendar_time_context)
        self.cross_calendar_time_weight = float(cross_calendar_time_weight)
        self.user_time_segment_experts = bool(user_time_segment_experts)
        self.segment_expert_weight = float(segment_expert_weight)
        self.calendar_domain_router = bool(calendar_domain_router)
        self.calendar_domain_router_weight = float(calendar_domain_router_weight)
        self.query_ranklift_regularizer = bool(query_ranklift_regularizer)
        self.query_ranklift_weight = float(query_ranklift_weight)
        self.query_ranklift_warmup_epoch = int(query_ranklift_warmup_epoch)
        self._last_q_tokens_for_ranklift: Optional[torch.Tensor] = None
        self._last_output_for_ranklift: Optional[torch.Tensor] = None
        self.user_time_film = bool(user_time_film)
        self.time_delta_sidecar = bool(time_delta_sidecar)
        self.user_time_evidence_block = bool(user_time_evidence_block)
        self.target_lite_domain_router = bool(target_lite_domain_router)
        self.low_card_temporal_content_sidecar = bool(low_card_temporal_content_sidecar)
        self.query_conditioned_time_attention = bool(query_conditioned_time_attention)
        self.query_time_rank = int(query_time_rank)
        self.user_time_item_mixer = bool(user_time_item_mixer)
        self.user_time_item_weight = float(user_time_item_weight)
        self.calendar_user_activity_cross = bool(calendar_user_activity_cross)
        self.head_recent_activity_cross = bool(head_recent_activity_cross)
        self.user_field_coverage_time_context = bool(user_field_coverage_time_context)
        self.dense_semantic_time_bilinear = bool(dense_semantic_time_bilinear)
        self.hrrm_dq_adapter = bool(hrrm_dq_adapter)

        # ================== NS Tokens Construction ==================

        if ns_tokenizer_type == 'group':
            # Original: one NS token per group
            self.user_ns_tokenizer = GroupNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = len(user_ns_groups)

            self.item_ns_tokenizer = GroupNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = len(item_ns_groups)
        elif ns_tokenizer_type == 'rankmixer':
            # RankMixer paper style: all embeddings cat → split → project
            # 0 means auto: fall back to group count
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
            )
            num_item_ns = item_ns_tokens
        elif ns_tokenizer_type == 'role_rankmixer':
            if user_ns_tokens <= 0:
                user_ns_tokens = len(user_ns_groups)
            if item_ns_tokens <= 0:
                item_ns_tokens = len(item_ns_groups)
            self.user_ns_tokenizer = RoleStratifiedRankMixerNSTokenizer(
                feature_specs=user_int_feature_specs,
                groups=user_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=user_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                pair_fids=role_pair_fids,
                role_low_card_threshold=role_low_card_threshold,
                role_high_card_threshold=role_high_card_threshold,
            )
            num_user_ns = user_ns_tokens

            self.item_ns_tokenizer = RoleStratifiedRankMixerNSTokenizer(
                feature_specs=item_int_feature_specs,
                groups=item_ns_groups,
                emb_dim=emb_dim,
                d_model=d_model,
                num_ns_tokens=item_ns_tokens,
                emb_skip_threshold=emb_skip_threshold,
                pair_fids=role_pair_fids,
                role_low_card_threshold=role_low_card_threshold,
                role_high_card_threshold=role_high_card_threshold,
            )
            num_item_ns = item_ns_tokens
        else:
            raise ValueError(f"Unknown ns_tokenizer_type: {ns_tokenizer_type}")

        # User dense feature projection (if available)
        self.has_user_dense = user_dense_dim > 0
        if self.has_user_dense:
            if dense_pair_compressor:
                self.user_dense_proj = DenseIntPairCompressor(
                    user_dense_dim=user_dense_dim,
                    user_dense_feature_specs=user_dense_feature_specs,
                    user_int_feature_specs=user_int_feature_specs,
                    emb_dim=emb_dim,
                    d_model=d_model,
                    emb_skip_threshold=emb_skip_threshold,
                    pair_fids=dense_pair_fids,
                    high_dim_threshold=dense_pair_high_dim_threshold,
                    residual_weight=dense_pair_weight,
                )
            else:
                self.user_dense_proj = nn.Sequential(
                    nn.Linear(user_dense_dim, d_model),
                    nn.LayerNorm(d_model),
                )
        self.uses_dense_pair_compressor = self.has_user_dense and bool(dense_pair_compressor)

        # Item dense feature projection (if available)
        self.has_item_dense = item_dense_dim > 0
        if self.has_item_dense:
            self.item_dense_proj = nn.Sequential(
                nn.Linear(item_dense_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # Total NS token count
        self.num_ns = (num_user_ns + (1 if self.has_user_dense else 0)
                       + num_item_ns + (1 if self.has_item_dense else 0))

        # ================== Check d_model % T == 0 constraint (full mode only) ==================
        T = num_queries * self.num_sequences + self.num_ns
        if rank_mixer_mode == 'full' and d_model % T != 0:
            valid_T_values = [t for t in range(1, d_model + 1) if d_model % t == 0]
            raise ValueError(
                f"d_model={d_model} must be divisible by T=num_queries*num_sequences+num_ns="
                f"{num_queries}*{self.num_sequences}+{self.num_ns}={T}. "
                f"Valid T values for d_model={d_model}: {valid_T_values}"
            )

        # ================== Seq Tokens Embedding ==================
        # seq_id_threshold decides which features inside the seq tokenizer are
        # treated as id features (they receive extra dropout). It is fully
        # independent of emb_skip_threshold (which skips Embedding creation).
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        def _make_seq_embs(vocab_sizes):
            """Create embedding list, returning None for features skipped via
            emb_skip_threshold or with no vocab info (vs<=0)."""
            embs_raw = []
            for vs in vocab_sizes:
                skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
                if skip:
                    embs_raw.append(None)
                else:
                    embs_raw.append(nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0))
            module_list = nn.ModuleList([e for e in embs_raw if e is not None])
            # Map from position index to real index in module_list (-1 if skipped)
            index_map = []
            real_idx = 0
            for e in embs_raw:
                if e is not None:
                    index_map.append(real_idx)
                    real_idx += 1
                else:
                    index_map.append(-1)
            is_id = [int(vs) > seq_id_threshold for vs in vocab_sizes]
            return module_list, index_map, is_id

        # ================== Dynamic Sequence Embeddings ==================
        self._seq_embs = nn.ModuleDict()
        self._seq_emb_index = {}    # domain -> index_map
        self._seq_is_id = {}        # domain -> is_id list
        self._seq_vocab_sizes = {}  # domain -> vocab_sizes list
        self._seq_proj = nn.ModuleDict()
        self._seq_role_indices: Dict[str, Dict[str, List[int]]] = {}
        self._seq_low_proj = nn.ModuleDict()
        self._seq_stat_proj = nn.ModuleDict()
        self._seq_id_proj = nn.ModuleDict()

        for domain in self.seq_domains:
            vs = seq_vocab_sizes[domain]
            embs, idx_map, is_id = _make_seq_embs(vs)
            self._seq_embs[domain] = embs
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id
            self._seq_vocab_sizes[domain] = vs
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(len(vs) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            low_idx = [i for i, v in enumerate(vs) if 0 < int(v) <= self.seq_low_card_threshold]
            stat_idx = [i for i, v in enumerate(vs)
                        if self.seq_low_card_threshold < int(v) <= self.seq_id_threshold]
            id_idx = [i for i, v in enumerate(vs) if int(v) > self.seq_id_threshold]
            self._seq_role_indices[domain] = {
                'low': low_idx,
                'stat': stat_idx,
                'id': id_idx,
            }
            self._seq_low_proj[domain] = nn.Sequential(
                nn.Linear(max(1, len(low_idx)) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            self._seq_stat_proj[domain] = nn.Sequential(
                nn.Linear(max(1, len(stat_idx)) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            self._seq_id_proj[domain] = nn.Sequential(
                nn.Linear(max(1, len(id_idx)) * emb_dim, d_model),
                nn.LayerNorm(d_model),
            )
            if self.semantic_seq_tokenizer:
                logging.info(
                    f"SemanticSeqTokenizer {domain}: "
                    f"low={len(low_idx)}, stat={len(stat_idx)}, id={len(id_idx)}, "
                    f"id_weight={self.seq_high_card_residual_weight}"
                )

        # ================== Time Interval Bucket Embedding (optional) ==================
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, d_model, padding_idx=0)

        # ================== HyFormer Components ==================
        # MultiSeqQueryGenerator
        query_extra_tokens = 1 if (
            self.use_prequery_role_context
            or self.item_crossnet
            or self.exposure_time_context
            or self.user_time_segment_experts
            or self.time_delta_sidecar
            or self.user_time_evidence_block
            or self.target_lite_domain_router
            or self.low_card_temporal_content_sidecar
            or self.user_time_item_mixer
            or self.calendar_user_activity_cross
            or self.head_recent_activity_cross
            or self.user_field_coverage_time_context
            or self.dense_semantic_time_bilinear
        ) else 0
        self.query_generator = MultiSeqQueryGenerator(
            d_model=d_model,
            num_ns=self.num_ns,
            num_queries=num_queries,
            num_sequences=self.num_sequences,
            hidden_mult=hidden_mult,
            extra_context_tokens=query_extra_tokens,
        )

        if self.use_prequery_role_context:
            self.prequery_context_proj = nn.Sequential(
                nn.LayerNorm(d_model * 4),
                nn.Linear(d_model * 4, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            logging.info(
                f"PreQuery role context enabled: weight={self.prequery_context_weight}"
            )

        if self.item_crossnet:
            self.item_crossnet_block = LowRankCrossNet(
                d_model=d_model,
                rank=item_cross_rank,
                layers=2,
            )
            logging.info(
                f"Item-conditioned DensePair CrossNet enabled: "
                f"rank={item_cross_rank}, weight={self.item_cross_weight}"
            )

        if self.exposure_time_context:
            exposure_weight = (
                cross_calendar_time_weight
                if self.cross_calendar_time_context
                else exposure_time_weight
            )
            self.exposure_time_block = ExposureTimeContext(
                d_model=d_model,
                weight=exposure_weight,
                multi_resolution=multi_res_exposure_time,
                calendar_embeddings=calendar_time_embeddings,
                cross_calendar_time_context=cross_calendar_time_context,
            )
            logging.info(
                "Exposure user-time context enabled: "
                f"weight={exposure_weight}, multi_res={multi_res_exposure_time}, "
                f"calendar_emb={calendar_time_embeddings}, "
                f"cross_calendar={cross_calendar_time_context}"
            )

        if self.user_time_segment_experts:
            self.user_time_segment_block = UserTimeSegmentExpertContext(
                d_model=d_model,
                weight=segment_expert_weight,
                multi_resolution=multi_res_exposure_time,
            )
            logging.info(
                f"User-time segment experts enabled: weight={segment_expert_weight}"
            )

        if self.calendar_user_activity_cross:
            self.calendar_user_activity_block = CalendarUserActivityCrossContext(
                d_model=d_model,
                num_domains=self.num_sequences,
                weight=calendar_user_activity_weight,
            )
            logging.info(
                "Calendar x user-activity cross context enabled: "
                f"weight={calendar_user_activity_weight}"
            )

        if self.head_recent_activity_cross:
            self.head_recent_activity_block = HeadRecentActivityCrossContext(
                d_model=d_model,
                num_domains=self.num_sequences,
                weight=head_recent_activity_weight,
                head_k=head_recent_activity_k,
            )
            logging.info(
                "Head-recent activity x calendar context enabled: "
                f"weight={head_recent_activity_weight}, head_k={head_recent_activity_k}"
            )

        if self.user_field_coverage_time_context:
            self.user_field_coverage_time_block = UserFieldCoverageTimeContext(
                d_model=d_model,
                user_int_feature_specs=user_int_feature_specs,
                num_domains=self.num_sequences,
                weight=user_field_coverage_time_weight,
            )
            logging.info(
                "User field coverage x time context enabled: "
                f"weight={user_field_coverage_time_weight}"
            )

        if self.dense_semantic_time_bilinear:
            self.dense_semantic_time_block = DenseSemanticTimeBilinearContext(
                d_model=d_model,
                rank=dense_semantic_time_rank,
                weight=dense_semantic_time_weight,
            )
            logging.info(
                "Dense semantic time-bilinear context enabled: "
                f"rank={dense_semantic_time_rank}, weight={dense_semantic_time_weight}"
            )

        if self.calendar_domain_router:
            self.calendar_domain_router_block = CalendarDomainRouter(
                d_model=d_model,
                num_domains=self.num_sequences,
                weight=calendar_domain_router_weight,
                multi_resolution=multi_res_exposure_time,
            )
            logging.info(
                f"Calendar-conditioned domain router enabled: "
                f"weight={calendar_domain_router_weight}"
            )

        if self.query_ranklift_regularizer:
            logging.info(
                "Query RankLift regularizer enabled: "
                f"weight={query_ranklift_weight}, warmup_epoch={query_ranklift_warmup_epoch}"
            )

        if self.user_time_film:
            self.user_time_film_block = UserTimeFiLMModulator(
                d_model=d_model,
                weight=user_time_film_weight,
                multi_resolution=multi_res_exposure_time,
            )
            logging.info(
                "User-time FiLM enabled: "
                f"weight={user_time_film_weight}, multi_res={multi_res_exposure_time}"
            )

        if self.time_delta_sidecar:
            self.time_delta_sidecar_block = TimeDeltaHistogramSidecar(
                d_model=d_model,
                num_domains=self.num_sequences,
                weight=time_delta_sidecar_weight,
            )
            logging.info(
                f"Time-delta histogram sidecar enabled: weight={time_delta_sidecar_weight}"
            )

        if self.user_time_evidence_block:
            self.user_time_evidence_block_mod = UserTimeEvidenceRecBlock(
                d_model=d_model,
                weight=user_time_evidence_weight,
                multi_resolution=multi_res_exposure_time,
                calendar_embeddings=calendar_time_embeddings,
            )
            logging.info(
                "Unified user-time evidence RecBlock enabled: "
                f"weight={user_time_evidence_weight}, multi_res={multi_res_exposure_time}, "
                f"calendar_emb={calendar_time_embeddings}"
            )

        if self.target_lite_domain_router:
            self.target_lite_router = TargetLiteDomainRouter(
                d_model=d_model,
                emb_dim=emb_dim,
                item_int_feature_specs=item_int_feature_specs,
                item_dense_dim=item_dense_dim,
                num_domains=self.num_sequences,
                low_card_threshold=target_lite_low_card_threshold,
                weight=target_lite_weight,
            )
            logging.info(
                "Target-lite user-time domain router enabled: "
                f"low_card_threshold={target_lite_low_card_threshold}, "
                f"weight={target_lite_weight}, "
                f"low_card_fields={len(self.target_lite_router.low_card_indices)}"
            )

        if self.low_card_temporal_content_sidecar:
            self.low_card_content_proj = nn.ModuleDict({
                domain: nn.Sequential(
                    nn.LayerNorm(emb_dim),
                    nn.Linear(emb_dim, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model),
                )
                for domain in self.seq_domains
            })
            self.low_card_content_sidecar_block = LowCardTemporalContentSidecar(
                d_model=d_model,
                num_domains=self.num_sequences,
                weight=low_card_content_weight,
            )
            logging.info(
                "Low-card temporal content sidecar enabled: "
                f"weight={low_card_content_weight}, windows=4"
            )

        if self.user_time_item_mixer:
            self.user_time_item_block = UserTimeItemEvidenceMixer(
                d_model=d_model,
                rank=user_time_item_rank,
                weight=user_time_item_weight,
            )
            logging.info(
                f"User-time-item evidence mixer enabled: "
                f"rank={user_time_item_rank}, weight={user_time_item_weight}"
            )

        if self.time_attention_bias:
            bias_cls = (
                QueryConditionedTimeAttentionBias
                if self.query_conditioned_time_attention
                else TimeAttentionBias
            )
            if self.query_conditioned_time_attention:
                self.time_bias_modules = nn.ModuleDict({
                    domain: bias_cls(
                        d_model=d_model,
                        num_heads=num_heads,
                        num_time_buckets=max(1, num_time_buckets),
                        rank=query_time_rank,
                        clip=self.time_attention_clip,
                    )
                    for domain in self.seq_domains
                })
            else:
                self.time_bias_modules = nn.ModuleDict({
                    domain: bias_cls(
                        num_time_buckets=max(1, num_time_buckets),
                        hidden=16,
                        clip=self.time_attention_clip,
                    )
                    for domain in self.seq_domains
                })
            logging.info(
                "Time-aware CrossAttention bias enabled: "
                f"clip={self.time_attention_clip}, "
                f"query_conditioned={self.query_conditioned_time_attention}"
            )

        if self.hrrm_dq_adapter:
            self.hrrm_dq = MinimalHRRMDQAdapter(
                d_model=d_model,
                hidden_mult=2,
                recent_k=hrrm_recent_k,
                query_weight=hrrm_query_weight,
                memory_weight=hrrm_memory_weight,
            )
            logging.info(
                f"Minimal HRRM-DQ adapter enabled: query_weight={hrrm_query_weight}, "
                f"memory_weight={hrrm_memory_weight}, recent_k={hrrm_recent_k}"
            )

        # MultiSeqHyFormerBlock stack
        self.blocks = nn.ModuleList([
            MultiSeqHyFormerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_queries=num_queries,
                num_ns=self.num_ns,
                num_sequences=self.num_sequences,
                seq_encoder_type=seq_encoder_type,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                top_k=seq_top_k,
                causal=seq_causal,
                rank_mixer_mode=rank_mixer_mode,
            )
            for _ in range(num_hyformer_blocks)
        ])

        # ================== RoPE ==================
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, base=rope_base)
        else:
            self.rotary_emb = None

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(num_queries * self.num_sequences * d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Dropout
        self.emb_dropout = nn.Dropout(dropout_rate)

        # Classifier
        self.clsfier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num)
        )

        # Initialize parameters
        self._init_params()

        # Log emb_skip_threshold filtering stats
        if emb_skip_threshold > 0:
            def _count_filtered(vocab_sizes, emb_index):
                filtered = sum(1 for idx in emb_index if idx == -1)
                return filtered, len(vocab_sizes)
            for domain in self.seq_domains:
                f, t = _count_filtered(self._seq_vocab_sizes[domain], self._seq_emb_index[domain])
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {domain} skipped {f}/{t} features")
            for name, tokenizer in [
                ("user_ns", self.user_ns_tokenizer),
                ("item_ns", self.item_ns_tokenizer),
            ]:
                f = sum(1 for idx in tokenizer._emb_index if idx == -1)
                t = len(tokenizer._emb_index)
                if f > 0:
                    logging.info(f"emb_skip_threshold={emb_skip_threshold}: {name} skipped {f}/{t} features")

    def _init_params(self) -> None:
        """Applies Xavier initialization to all embedding weights."""
        for domain in self.seq_domains:
            for emb in self._seq_embs[domain]:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        for tokenizer in [self.user_ns_tokenizer, self.item_ns_tokenizer]:
            for emb in tokenizer.embs:
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0

        if getattr(self, 'uses_dense_pair_compressor', False):
            self.user_dense_proj.init_embeddings()

        if self.num_time_buckets > 0:
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0

    def reinit_high_cardinality_params(
        self, cardinality_threshold: int = 10000
    ) -> "set[int]":
        """Reinitializes embeddings whose vocab size exceeds the threshold.

        In this competition setup, threshold=0 intentionally cold-restarts all
        sparse feature embeddings with vocab_size > 0 at each epoch end. This
        preserves the strong v25.2 sparse regularization behavior.

        Args:
            cardinality_threshold: Only embeddings with vocab_size exceeding
                this value are reinitialized.

        Returns:
            A set of data_ptr() values for reinitialized parameters.
        """
        reinit_count = 0
        skip_count = 0
        reinit_ptrs = set()

        for emb_list, vocab_sizes, emb_index in [
            (self._seq_embs[d], self._seq_vocab_sizes[d], self._seq_emb_index[d])
            for d in self.seq_domains
        ]:
            for i, vs in enumerate(vocab_sizes):
                real_idx = emb_index[i]
                if real_idx == -1:
                    # Skipped by emb_skip_threshold, no embedding to reinit
                    continue
                emb = emb_list[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        for tokenizer, specs in [
            (self.user_ns_tokenizer, self.user_ns_tokenizer.feature_specs),
            (self.item_ns_tokenizer, self.item_ns_tokenizer.feature_specs),
        ]:
            for i, spec in enumerate(specs):
                _, vs, _, _ = _unpack_feature_spec(spec)
                real_idx = tokenizer._emb_index[i]
                if real_idx == -1:
                    continue
                emb = tokenizer.embs[real_idx]
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        if getattr(self, 'target_lite_domain_router', False):
            for emb, idx in zip(
                self.target_lite_router.low_card_embs,
                self.target_lite_router.low_card_indices,
            ):
                _, vs, _, _ = _unpack_feature_spec(
                    self.target_lite_router.item_int_feature_specs[idx])
                if int(vs) > cardinality_threshold:
                    nn.init.xavier_normal_(emb.weight.data)
                    emb.weight.data[0, :] = 0
                    reinit_ptrs.add(emb.weight.data_ptr())
                    reinit_count += 1
                else:
                    skip_count += 1

        # time_embedding is always preserved
        if self.num_time_buckets > 0:
            skip_count += 1

        logging.info(
            f"Re-initialized {reinit_count} sparse Embeddings "
            f"(vocab>{cardinality_threshold}; threshold=0 means reset all vocab>0), "
            f"kept {skip_count}"
        )
        return reinit_ptrs

    def get_sparse_params(self) -> List[nn.Parameter]:
        """Returns all embedding table parameters (optimized with Adagrad)."""
        sparse_params = set()
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                sparse_params.add(module.weight.data_ptr())
        return [p for p in self.parameters() if p.data_ptr() in sparse_params]

    def get_dense_params(self) -> List[nn.Parameter]:
        """Returns all non-embedding parameters (optimized with AdamW)."""
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    def _embed_seq_domain(
        self,
        domain: str,
        seq: torch.Tensor,
        sideinfo_embs: nn.ModuleList,
        proj: nn.Module,
        is_id: List[bool],
        emb_index: List[int],
        time_bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embeds a sequence domain by concatenating sideinfo embeddings and projecting to d_model."""
        B, S, L = seq.shape
        emb_list = []
        for i in range(S):
            real_idx = emb_index[i] if i < len(emb_index) else -1
            if real_idx == -1:
                # Feature skipped by emb_skip_threshold: output zero vector
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                emb = sideinfo_embs[real_idx]
                e = emb(seq[:, i, :])  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)

        if self.semantic_seq_tokenizer:
            return self._embed_seq_domain_semantic(domain, emb_list, time_bucket_ids)

        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, S*emb_dim)
        token_emb = F.gelu(proj(cat_emb))  # (B, L, D)

        # Add time bucket embedding (all-zero ids produce zero vectors via padding_idx=0)
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)

        return token_emb

    def _cat_seq_role(self, domain: str, emb_list: List[torch.Tensor], role: str) -> torch.Tensor:
        indices = self._seq_role_indices[domain][role]
        B, L, _ = emb_list[0].shape
        if not indices:
            return emb_list[0].new_zeros(B, L, self.emb_dim)
        return torch.cat([emb_list[i] for i in indices], dim=-1)

    def _embed_seq_domain_semantic(
        self,
        domain: str,
        emb_list: List[torch.Tensor],
        time_bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        low_cat = self._cat_seq_role(domain, emb_list, 'low')
        stat_cat = self._cat_seq_role(domain, emb_list, 'stat')
        id_cat = self._cat_seq_role(domain, emb_list, 'id')

        low_part = F.gelu(self._seq_low_proj[domain](low_cat))
        stat_part = F.gelu(self._seq_stat_proj[domain](stat_cat))
        id_part = torch.tanh(self._seq_id_proj[domain](id_cat))

        token_emb = low_part + stat_part + self.seq_high_card_residual_weight * id_part
        if self.num_time_buckets > 0:
            token_emb = token_emb + self.time_embedding(time_bucket_ids)
        return F.gelu(token_emb)

    def _masked_mean(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        valid = (~padding_mask).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1)
        return (tokens * valid).sum(dim=1) / denom

    def _build_prequery_context(
        self,
        user_ns: torch.Tensor,
        item_ns: torch.Tensor,
        user_dense_tok: Optional[torch.Tensor],
        seq_tokens_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
    ) -> torch.Tensor:
        user_ctx = user_ns.mean(dim=1)
        item_ctx = item_ns.mean(dim=1)
        if user_dense_tok is None:
            dense_ctx = user_ctx.new_zeros(user_ctx.shape)
        else:
            dense_ctx = user_dense_tok.squeeze(1)
        seq_ctx = torch.stack([
            self._masked_mean(tokens, mask)
            for tokens, mask in zip(seq_tokens_list, seq_masks_list)
        ], dim=1).mean(dim=1)
        ctx = self.prequery_context_proj(torch.cat([user_ctx, item_ctx, dense_ctx, seq_ctx], dim=-1))
        return self.prequery_context_weight * ctx

    def _target_context(
        self,
        item_ns: torch.Tensor,
        item_dense_tok: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if item_dense_tok is None:
            return item_ns.mean(dim=1)
        return torch.cat([item_ns, item_dense_tok], dim=1).mean(dim=1)

    def _build_low_card_temporal_content_context(
        self,
        seq_data: Optional[Dict[str, torch.Tensor]],
        seq_masks_list: List[torch.Tensor],
        seq_time_buckets: Optional[Dict[str, torch.Tensor]],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if seq_data is None or seq_time_buckets is None:
            B = ref.shape[0]
            token_count = self.num_sequences * 4
            return self.low_card_content_sidecar_block(
                ref.new_zeros(B, token_count, self.d_model))

        all_window_tokens = []
        edges = (21, 31, 47, 64)
        for domain, mask in zip(self.seq_domains, seq_masks_list):
            seq = seq_data[domain]
            tb = seq_time_buckets.get(domain)
            content_indices = (
                self._seq_role_indices[domain]['low']
                + self._seq_role_indices[domain]['stat']
            )
            content_embs = []
            B, _, L = seq.shape
            for idx in content_indices:
                real_idx = self._seq_emb_index[domain][idx]
                if real_idx == -1:
                    continue
                emb = self._seq_embs[domain][real_idx]
                vals = seq[:, idx, :].long()
                e = emb(vals)
                valid_value = (vals > 0).float().unsqueeze(-1)
                content_embs.append(e * valid_value)
            if content_embs:
                low_content = torch.stack(content_embs, dim=0).mean(dim=0)
                low_content = self.low_card_content_proj[domain](low_content)
            else:
                low_content = ref.new_zeros(B, L, self.d_model)

            if tb is None:
                all_window_tokens.extend([ref.new_zeros(B, self.d_model) for _ in edges])
                continue
            valid = ((~mask) & (tb > 0)).float()
            prev = 0
            for edge in edges:
                win = ((tb > prev) & (tb <= edge)).float() * valid
                denom = win.sum(dim=1, keepdim=True).clamp(min=1.0)
                pooled = (low_content * win.unsqueeze(-1)).sum(dim=1) / denom
                all_window_tokens.append(pooled)
                prev = edge
        content_tokens = torch.stack(all_window_tokens, dim=1)
        return self.low_card_content_sidecar_block(content_tokens)

    def _build_item_cross_context(
        self,
        user_ns: torch.Tensor,
        item_ns: torch.Tensor,
        user_dense_tok: Optional[torch.Tensor],
        item_dense_tok: Optional[torch.Tensor],
    ) -> torch.Tensor:
        user_ctx = user_ns.mean(dim=1)
        item_ctx = self._target_context(item_ns, item_dense_tok)
        if user_dense_tok is None:
            dense_ctx = user_ctx.new_zeros(user_ctx.shape)
        else:
            dense_ctx = user_dense_tok.squeeze(1)
        return self.item_cross_weight * self.item_crossnet_block(
            dense_ctx=dense_ctx,
            item_ctx=item_ctx,
            user_ctx=user_ctx,
        )

    def _build_extra_query_context(
        self,
        user_ns: torch.Tensor,
        item_ns: torch.Tensor,
        user_dense_tok: Optional[torch.Tensor],
        item_dense_tok: Optional[torch.Tensor],
        seq_tokens_list: List[torch.Tensor],
        seq_masks_list: List[torch.Tensor],
        user_int_feats: Optional[torch.Tensor] = None,
        user_dense_feats: Optional[torch.Tensor] = None,
        item_int_feats: Optional[torch.Tensor] = None,
        item_dense_feats: Optional[torch.Tensor] = None,
        seq_data: Optional[Dict[str, torch.Tensor]] = None,
        seq_time_buckets: Optional[Dict[str, torch.Tensor]] = None,
        timestamp: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        contexts = []
        if self.use_prequery_role_context:
            contexts.append(self._build_prequery_context(
                user_ns, item_ns, user_dense_tok, seq_tokens_list, seq_masks_list))
        if self.item_crossnet:
            contexts.append(self._build_item_cross_context(
                user_ns, item_ns, user_dense_tok, item_dense_tok))
        user_ctx = user_ns.mean(dim=1)
        item_ctx = self._target_context(item_ns, item_dense_tok)
        if user_dense_tok is None:
            dense_ctx = user_ctx.new_zeros(user_ctx.shape)
        else:
            dense_ctx = user_dense_tok.squeeze(1)
        if self.exposure_time_context:
            contexts.append(self.exposure_time_block(timestamp, dense_ctx, user_ctx))
        if self.user_time_segment_experts:
            contexts.append(self.user_time_segment_block(timestamp, dense_ctx, user_ctx))
        if self.calendar_user_activity_cross:
            contexts.append(self.calendar_user_activity_block(
                timestamp, dense_ctx, user_ctx, seq_time_buckets,
                seq_masks_list, self.seq_domains))
        if self.head_recent_activity_cross:
            contexts.append(self.head_recent_activity_block(
                timestamp, dense_ctx, user_ctx, seq_time_buckets,
                seq_masks_list, self.seq_domains))
        if self.user_field_coverage_time_context and user_int_feats is not None:
            contexts.append(self.user_field_coverage_time_block(
                timestamp, dense_ctx, user_ctx, user_int_feats,
                seq_time_buckets, seq_masks_list, self.seq_domains))
        if (
            self.dense_semantic_time_bilinear
            and self.uses_dense_pair_compressor
            and user_int_feats is not None
            and user_dense_feats is not None
            and isinstance(self.user_dense_proj, DenseIntPairCompressor)
        ):
            semantic_ctx, scalar_ctx, pair_ctx = self.user_dense_proj.component_contexts(
                user_dense_feats, user_int_feats)
            contexts.append(self.dense_semantic_time_block(
                timestamp, semantic_ctx, scalar_ctx, pair_ctx))
        sidecar_ctx = None
        if self.time_delta_sidecar and seq_time_buckets is not None:
            sidecar_ctx = self.time_delta_sidecar_block(
                seq_time_buckets, seq_masks_list, self.seq_domains, dense_ctx)
            if not self.user_time_evidence_block:
                contexts.append(sidecar_ctx)
        if self.low_card_temporal_content_sidecar:
            low_card_ctx = self._build_low_card_temporal_content_context(
                seq_data, seq_masks_list, seq_time_buckets, dense_ctx)
            if self.user_time_evidence_block:
                sidecar_ctx = low_card_ctx if sidecar_ctx is None else sidecar_ctx + low_card_ctx
            else:
                contexts.append(low_card_ctx)
        if self.user_time_evidence_block:
            contexts.append(self.user_time_evidence_block_mod(
                timestamp, dense_ctx, user_ctx, sidecar_ctx))
        if self.target_lite_domain_router and item_int_feats is not None:
            contexts.append(self.target_lite_router(
                item_int_feats, item_dense_feats, dense_ctx, user_ctx,
                seq_time_buckets, seq_masks_list, self.seq_domains))
        if self.user_time_item_mixer:
            contexts.append(self.user_time_item_block(timestamp, dense_ctx, user_ctx, item_ctx))
        if not contexts:
            return None
        return torch.stack(contexts, dim=0).sum(dim=0)

    def _user_dense_context(
        self,
        user_ns: torch.Tensor,
        user_dense_tok: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_ctx = user_ns.mean(dim=1)
        if user_dense_tok is None:
            dense_ctx = user_ctx.new_zeros(user_ctx.shape)
        else:
            dense_ctx = user_dense_tok.squeeze(1)
        return dense_ctx, user_ctx

    def _apply_calendar_domain_router(
        self,
        q_tokens_list: List[torch.Tensor],
        timestamp: Optional[torch.Tensor],
        user_ns: torch.Tensor,
        user_dense_tok: Optional[torch.Tensor],
        seq_time_buckets: Optional[Dict[str, torch.Tensor]],
        seq_masks_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        if not self.calendar_domain_router:
            return q_tokens_list
        dense_ctx, user_ctx = self._user_dense_context(user_ns, user_dense_tok)
        deltas = self.calendar_domain_router_block(
            timestamp, dense_ctx, user_ctx, seq_time_buckets,
            seq_masks_list, self.seq_domains)
        return [
            q_tokens + deltas[:, idx:idx + 1, :]
            for idx, q_tokens in enumerate(q_tokens_list)
        ]

    @staticmethod
    def _effective_rank_for_logging(tokens: torch.Tensor) -> float:
        with torch.no_grad():
            x = tokens.detach().float()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            if x.shape[0] <= 1:
                return 0.0
            x = x - x.mean(dim=0, keepdim=True)
            s = torch.linalg.svdvals(x)
            denom = s.sum().clamp(min=1.0e-8)
            p = s / denom
            rank = torch.exp(-(p * torch.log(p.clamp(min=1.0e-8))).sum())
            return float(rank.detach().cpu().item())

    def query_ranklift_regularization_loss(self) -> Optional[torch.Tensor]:
        q_tokens = self._last_q_tokens_for_ranklift
        if q_tokens is None:
            return None
        B, T, _ = q_tokens.shape
        if T <= 1:
            return q_tokens.new_zeros(())
        q_norm = F.normalize(q_tokens, dim=-1)
        sim = torch.bmm(q_norm, q_norm.transpose(1, 2))
        eye = torch.eye(T, device=q_tokens.device, dtype=q_tokens.dtype).unsqueeze(0)
        off_diag = sim * (1.0 - eye)
        return (off_diag ** 2).sum() / float(B * T * (T - 1))

    def query_ranklift_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if self._last_q_tokens_for_ranklift is not None:
            metrics["q_tokens_effective_rank"] = self._effective_rank_for_logging(
                self._last_q_tokens_for_ranklift)
        if self._last_output_for_ranklift is not None:
            metrics["output_effective_rank"] = self._effective_rank_for_logging(
                self._last_output_for_ranklift)
        return metrics

    def _build_time_attention_biases(
        self,
        seq_time_buckets: Dict[str, torch.Tensor],
        seq_masks_list: List[torch.Tensor],
        q_tokens_list: Optional[List[torch.Tensor]] = None,
    ) -> Optional[List[Optional[torch.Tensor]]]:
        if not self.time_attention_bias:
            return None
        out = []
        for idx, (domain, mask) in enumerate(zip(self.seq_domains, seq_masks_list)):
            if self.query_conditioned_time_attention:
                if q_tokens_list is None:
                    raise ValueError("q_tokens_list is required for query-conditioned time attention")
                out.append(self.time_bias_modules[domain](
                    q_tokens_list[idx], seq_time_buckets[domain], mask))
            else:
                out.append(self.time_bias_modules[domain](seq_time_buckets[domain], mask))
        return out

    def _make_padding_mask(
        self, seq_len: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """Generates a padding mask from sequence lengths."""
        device = seq_len.device
        idx = torch.arange(max_len, device=device).unsqueeze(0)  # (1, max_len)
        return idx >= seq_len.unsqueeze(1)  # (B, max_len)

    def _run_multi_seq_blocks(
        self,
        q_tokens_list: list,
        ns_tokens: torch.Tensor,
        seq_tokens_list: list,
        seq_masks_list: list,
        attn_bias_list: Optional[List[Optional[torch.Tensor]]] = None,
        apply_dropout: bool = True
    ) -> torch.Tensor:
        """Runs the multi-sequence block stack with dropout and output projection."""
        if apply_dropout:
            q_tokens_list = [self.emb_dropout(q) for q in q_tokens_list]
            ns_tokens = self.emb_dropout(ns_tokens)
            seq_tokens_list = [self.emb_dropout(s) for s in seq_tokens_list]

        curr_qs = q_tokens_list
        curr_ns = ns_tokens
        curr_seqs = seq_tokens_list
        curr_masks = seq_masks_list

        for block in self.blocks:
            # Precompute RoPE cos/sin for each sequence
            rope_cos_list = None
            rope_sin_list = None
            if self.rotary_emb is not None:
                rope_cos_list = []
                rope_sin_list = []
                device = curr_seqs[0].device
                for seq_i in curr_seqs:
                    seq_len = seq_i.shape[1]
                    cos, sin = self.rotary_emb(seq_len, device)
                    rope_cos_list.append(cos)
                    rope_sin_list.append(sin)

            curr_qs, curr_ns, curr_seqs, curr_masks = block(
                q_tokens_list=curr_qs,
                ns_tokens=curr_ns,
                seq_tokens_list=curr_seqs,
                seq_padding_masks=curr_masks,
                rope_cos_list=rope_cos_list,
                rope_sin_list=rope_sin_list,
                attn_bias_list=attn_bias_list,
            )

        # Output: concatenate all sequences' Q tokens then project via MLP
        B = curr_qs[0].shape[0]
        all_q = torch.cat(curr_qs, dim=1)  # (B, Nq*S, D)
        output = all_q.view(B, -1)  # (B, Nq*S*D)
        output = self.output_proj(output)  # (B, D)

        return output

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        """Runs the forward pass of the PCVRHyFormer model."""
        # 1. NS tokens: grouped projection
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)   # (B, num_user_groups, D)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)   # (B, num_item_groups, D)

        ns_parts = [user_ns]
        user_dense_tok = None
        item_dense_tok = None
        if self.has_user_dense:
            if self.uses_dense_pair_compressor:
                user_dense_tok = self.user_dense_proj(
                    inputs.user_dense_feats, inputs.user_int_feats).unsqueeze(1)
            else:
                user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)
        if self.user_time_film:
            user_ns, user_dense_tok = self.user_time_film_block(
                inputs.timestamp, user_ns, user_dense_tok)
        if user_dense_tok is not None:
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)  # (B, 1, D)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)  # (B, num_ns, D)

        # 2. Embed each sequence domain (dynamic)
        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                domain,
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain])
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        # 3. Generate independent Q tokens per sequence via MultiSeqQueryGenerator
        extra_context = self._build_extra_query_context(
            user_ns, item_ns, user_dense_tok, item_dense_tok,
            seq_tokens_list, seq_masks_list,
            user_int_feats=inputs.user_int_feats,
            user_dense_feats=inputs.user_dense_feats,
            item_int_feats=inputs.item_int_feats,
            item_dense_feats=inputs.item_dense_feats,
            seq_data=inputs.seq_data,
            seq_time_buckets=inputs.seq_time_buckets,
            timestamp=inputs.timestamp)
        q_tokens_list = self.query_generator(
            ns_tokens, seq_tokens_list, seq_masks_list, extra_context=extra_context)
        q_tokens_list = self._apply_calendar_domain_router(
            q_tokens_list, inputs.timestamp, user_ns, user_dense_tok,
            inputs.seq_time_buckets, seq_masks_list)
        if self.hrrm_dq_adapter:
            target_ctx = self._target_context(item_ns, item_dense_tok)
            q_tokens_list = self.hrrm_dq(
                q_tokens_list, target_ctx, seq_tokens_list, seq_masks_list)
        if self.training and self.query_ranklift_regularizer:
            self._last_q_tokens_for_ranklift = torch.cat(q_tokens_list, dim=1)
        attn_bias_list = self._build_time_attention_biases(
            inputs.seq_time_buckets, seq_masks_list, q_tokens_list=q_tokens_list)

        # 4. Dropout + MultiSeqHyFormerBlock stack + output projection
        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            attn_bias_list=attn_bias_list,
            apply_dropout=self.training
        )
        if self.training and self.query_ranklift_regularizer:
            self._last_output_for_ranklift = output

        # 5. Classifier
        logits = self.clsfier(output)  # (B, action_num)
        return logits

    def predict(self, inputs: ModelInput) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs inference without dropout, returning both logits and embeddings."""
        # Reuses forward logic but without dropout
        user_ns = self.user_ns_tokenizer(inputs.user_int_feats)
        item_ns = self.item_ns_tokenizer(inputs.item_int_feats)

        ns_parts = [user_ns]
        user_dense_tok = None
        item_dense_tok = None
        if self.has_user_dense:
            if self.uses_dense_pair_compressor:
                user_dense_tok = self.user_dense_proj(
                    inputs.user_dense_feats, inputs.user_int_feats).unsqueeze(1)
            else:
                user_dense_tok = F.silu(self.user_dense_proj(inputs.user_dense_feats)).unsqueeze(1)
        if self.user_time_film:
            user_ns, user_dense_tok = self.user_time_film_block(
                inputs.timestamp, user_ns, user_dense_tok)
        if user_dense_tok is not None:
            ns_parts.append(user_dense_tok)
        ns_parts.append(item_ns)
        if self.has_item_dense:
            item_dense_tok = F.silu(self.item_dense_proj(inputs.item_dense_feats)).unsqueeze(1)
            ns_parts.append(item_dense_tok)

        ns_tokens = torch.cat(ns_parts, dim=1)

        seq_tokens_list = []
        seq_masks_list = []
        for domain in self.seq_domains:
            tokens = self._embed_seq_domain(
                domain,
                inputs.seq_data[domain],
                self._seq_embs[domain], self._seq_proj[domain],
                self._seq_is_id[domain], self._seq_emb_index[domain],
                inputs.seq_time_buckets[domain])
            seq_tokens_list.append(tokens)
            mask = self._make_padding_mask(inputs.seq_lens[domain], inputs.seq_data[domain].shape[2])
            seq_masks_list.append(mask)

        extra_context = self._build_extra_query_context(
            user_ns, item_ns, user_dense_tok, item_dense_tok,
            seq_tokens_list, seq_masks_list,
            user_int_feats=inputs.user_int_feats,
            user_dense_feats=inputs.user_dense_feats,
            item_int_feats=inputs.item_int_feats,
            item_dense_feats=inputs.item_dense_feats,
            seq_data=inputs.seq_data,
            seq_time_buckets=inputs.seq_time_buckets,
            timestamp=inputs.timestamp)
        q_tokens_list = self.query_generator(
            ns_tokens, seq_tokens_list, seq_masks_list, extra_context=extra_context)
        q_tokens_list = self._apply_calendar_domain_router(
            q_tokens_list, inputs.timestamp, user_ns, user_dense_tok,
            inputs.seq_time_buckets, seq_masks_list)
        if self.hrrm_dq_adapter:
            target_ctx = self._target_context(item_ns, item_dense_tok)
            q_tokens_list = self.hrrm_dq(
                q_tokens_list, target_ctx, seq_tokens_list, seq_masks_list)
        attn_bias_list = self._build_time_attention_biases(
            inputs.seq_time_buckets, seq_masks_list, q_tokens_list=q_tokens_list)

        output = self._run_multi_seq_blocks(
            q_tokens_list, ns_tokens, seq_tokens_list, seq_masks_list,
            attn_bias_list=attn_bias_list,
            apply_dropout=False
        )

        logits = self.clsfier(output)
        return logits, output
