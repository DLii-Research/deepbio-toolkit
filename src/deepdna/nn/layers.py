import abc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

# Multi-head Attention Mechanisms ------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout=0.0,
        bias=True,
        head_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        if head_embed_dim is None:
            assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads if head_embed_dim is not provided"
            head_embed_dim = embed_dim // num_heads
        self.head_embed_dim = head_embed_dim

        # Create projection layers
        self.w_query = nn.Linear(embed_dim, self.head_embed_dim * num_heads, bias=bias)
        self.w_key = nn.Linear(embed_dim, self.head_embed_dim * num_heads, bias=bias)
        self.w_value = nn.Linear(embed_dim, self.head_embed_dim * num_heads, bias=bias)
        self.w_output = nn.Linear(self.head_embed_dim * num_heads, embed_dim, bias=bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        *extra_dims_q, n_q, _ = query.size()
        *extra_dims_k, n_k, _ = key.size()
        *extra_dims_v, _, _ = value.size()

        # Linear projections
        q = self.w_query(query).view((*extra_dims_q, n_q, self.num_heads, self.head_embed_dim)).transpose(-2, -3) # [..., h, n_q, d_k]
        k = self.w_key(key).view((*extra_dims_k, n_k, self.num_heads, self.head_embed_dim)).transpose(-2, -3) # [..., h, n_k, d_k]
        v = self.w_value(value).view((*extra_dims_v, n_k, self.num_heads, self.head_embed_dim)).transpose(-2, -3) # [..., h, n_k, d_k]

        # Compute attention weights
        attention_weights = self.compute_attention_weights(q, k, self.merge_mask(attention_mask, key_padding_mask), attention_head_mask)

        attention = torch.matmul(attention_weights, v) # [..., h, n_q, d_k]
        attention = attention.transpose(-2, -3).reshape((*extra_dims_v, n_q, -1)) # [..., n_q, embed_dim]
        output = self.w_output(attention)

        if return_attention_weights:
            if average_attention_weights:
                n = attention_head_mask.sum() if attention_head_mask is not None else attention_weights.size(-1)
                attention_weights = attention_weights.sum(dim=-3) / n
            return output, attention_weights
        return output

    def merge_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor]
    ) -> Union[torch.Tensor, None]:
        if key_padding_mask is not None:
            key_padding_mask = torch.repeat_interleave(key_padding_mask.unsqueeze(-2), key_padding_mask.shape[-1], -2) # type: ignore
        if attention_mask is None:
            return key_padding_mask
        if key_padding_mask is None:
            return attention_mask
        return attention_mask | key_padding_mask

    def compute_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attention_head_mask: Optional[torch.Tensor]
    ):
        attention_weights = torch.matmul(query, key.transpose(-2, -1))/np.sqrt(self.head_embed_dim)
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask.unsqueeze(-3), float("-inf"))
        attention_weights = F.softmax(attention_weights, dim=-1) # [..., h, n_q, n_k]
        if attention_head_mask is not None:
            attention_weights = attention_weights * attention_head_mask.view((-1, 1, 1))
        return attention_weights


class RelativeMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_length: int,
        dropout=0.0,
        bias=True,
        head_embed_dim: Optional[int] = None,
    ):
        super().__init__(embed_dim, num_heads, dropout, bias, head_embed_dim)
        self.max_length = max_length
        self.Er = nn.Parameter(torch.randn(max_length, self.head_embed_dim))

    def _skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        *dims, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(*dims, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        s_rel = reshaped[(slice(None, None, None),)*len(dims) + (slice(1, None, None), slice(None, None))]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return s_rel

    def compute_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attention_head_mask: Optional[torch.Tensor]
    ):
        start = self.max_length - query.size(-2)
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(query, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self._skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        QK_t = torch.matmul(query, key.transpose(-2, -1))
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attention_weights = (QK_t + Srel) / np.sqrt(self.head_embed_dim)
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask.unsqueeze(-3), float("-inf"))
        attention_weights = F.softmax(attention_weights, dim=-1) # [..., h, n_q, n_k]
        if attention_head_mask is not None:
            attention_weights = attention_weights * attention_head_mask.view((-1, 1, 1))
        return attention_weights
