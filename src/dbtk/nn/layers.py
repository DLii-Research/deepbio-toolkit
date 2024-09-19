import abc
import copy
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, cast, List, Optional, Sequence, Tuple, TypeVar, Union

from .._utils import export

Mha = TypeVar("Mha", bound="MultiHeadAttention")

# Multi-head Attention Mechanisms ------------------------------------------------------------------

@export
class MultiHeadAttention(L.LightningModule):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        head_embed_dim: Optional[int] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
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
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        *extra_dims_q, n_q, _ = query.size()
        *extra_dims_k, n_k, _ = key.size()
        *extra_dims_v, _, _ = value.size()

        # Linear projections
        q = self.w_query(query).view((*extra_dims_q, n_q, self.num_heads, self.head_embed_dim)).transpose(-2, -3) # [..., h, n_q, d_k]
        k = self.w_key(key).view((*extra_dims_k, n_k, self.num_heads, self.head_embed_dim)).transpose(-2, -3) # [..., h, n_k, d_k]
        v = self.w_value(value).view((*extra_dims_v, n_k, self.num_heads, self.head_embed_dim)).transpose(-2, -3) # [..., h, n_k, d_k]

        # Compute attention weights
        attention_weights = self.compute_attention_weights(
            q,
            k,
            self.merge_mask(attention_mask, key_padding_mask),
            attention_head_mask)
        attention_weights = self.dropout(attention_weights)

        attention = torch.matmul(attention_weights, v) # [..., h, n_q, d_k]
        attention = attention.transpose(-2, -3).reshape((*extra_dims_v, n_q, self.num_heads*self.head_embed_dim)) # [..., n_q, d_k*num_heads]
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
            key_padding_mask = key_padding_mask.unsqueeze(-2)
            # key_padding_mask = torch.repeat_interleave(key_padding_mask.unsqueeze(-2), key_padding_mask.shape[-1], -2) # type: ignore
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
    ) -> torch.Tensor:
        attention_weights = torch.matmul(query, key.transpose(-2, -1))/np.sqrt(self.head_embed_dim)
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask.unsqueeze(-3), float("-inf"))
        attention_weights = F.softmax(attention_weights, dim=-1) # [..., h, n_q, n_k]
        if attention_head_mask is not None:
            attention_weights = attention_weights * attention_head_mask.view((-1, 1, 1))
        return attention_weights

    def __len__(self):
        return self.num_heads


@export
class RelativeMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_length: int,
        dropout: float = 0.0,
        bias: bool = True,
        head_embed_dim: Optional[int] = None
    ):
        super().__init__(embed_dim, num_heads, dropout, bias, head_embed_dim)
        self.max_length = max_length
        self.pos_embeddings = nn.Parameter(torch.randn(self.head_embed_dim, 2*max_length - 1)) # (dxn)

    def _skew(self, x: torch.Tensor):
        """
        Memory-efficient skew operation.
        """
        n = x.shape[-1] - x.shape[-1]//2
        x = F.pad(x, (0, 1))
        skewed = x.flatten(-2).narrow(-1, 0, x.shape[-2]*(x.shape[-1] - 1)).view((*x.shape[:-1], -1))
        rel = skewed.narrow(-1, n - 1, n)
        return rel

    def compute_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attention_head_mask: Optional[torch.Tensor]
    ):
        # Get required position embeddings
        n = key.shape[-2]
        pos_embeddings = F.pad(self.pos_embeddings, (n - self.max_length, n - self.max_length), mode="replicate")
        att_qk = torch.matmul(query, key.transpose(-2, -1))
        att_qrel = self._skew(torch.matmul(query, pos_embeddings))
        attention_weights = (att_qk + att_qrel) / np.sqrt(self.head_embed_dim)
        # att_qrel = self._skew(torch.matmul(query, self.pos_embeddings), key.shape[-2])
        # att_krel = self._skew(torch.matmul(key, self.pos_embeddings.flip(-1)), query.shape[-2]).transpose(-1, -2)
        # attention_weights = (att_qk + att_qrel + att_krel) / np.sqrt(self.head_embed_dim)
        if attention_mask is not None:
            attention_weights = attention_weights.masked_fill(attention_mask.unsqueeze(-3), float("-inf"))
        attention_weights = F.softmax(attention_weights, dim=-1) # [..., h, n_q, n_k]
        if attention_head_mask is not None:
            attention_weights = attention_weights * attention_head_mask.view((-1, 1, 1))
        return attention_weights

# Transformer Generics -----------------------------------------------------------------------------

@export
class MultiHeadAttentionBlock(L.LightningModule):
    def __init__(
        self,
        mha: MultiHeadAttention,
        feedforward_dim: int,
        feedforward_activation: Union[L.LightningModule, Callable] = F.gelu,
        norm_first: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.mha = mha
        self.feedforward_dim = feedforward_dim
        self.feedforward_activation = feedforward_activation
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(mha.embed_dim)
        self.norm2 = nn.LayerNorm(mha.embed_dim)
        self.feedforward_linear1 = nn.Linear(mha.embed_dim, feedforward_dim)
        self.feedforward_linear2 = nn.Linear(feedforward_dim, mha.embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def _cross_attention_block(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        attention_head_mask: Optional[torch.Tensor],
        average_attention_weights: bool,
        return_attention_weights: bool
    ):
        attention_output = self.mha(
            x,
            y,
            y,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            attention_head_mask=attention_head_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights)
        if isinstance(attention_output, tuple):
            attention_output, *extra = attention_output
        else:
            extra = None
        return self.dropout1(attention_output), extra

    def _feedforward_block(self, x: torch.Tensor):
        return self.dropout2(
            self.feedforward_linear2(self.feedforward_activation(self.feedforward_linear1(x)))
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        if self.norm_first:
            if x is y:
                x_norm = y_norm = self.norm1(x)
            else:
                x_norm = self.norm1(x)
                y_norm = self.norm1(y)
            attention_output, extra_output = self._cross_attention_block(
                x_norm,
                y_norm,
                attention_mask,
                key_padding_mask,
                attention_head_mask,
                average_attention_weights,
                return_attention_weights)
            x = x + attention_output
            x = x + self._feedforward_block(self.norm2(x))
        else:
            attention_output, extra_output = self._cross_attention_block(
                x,
                y,
                attention_mask,
                key_padding_mask,
                attention_head_mask,
                average_attention_weights,
                return_attention_weights)
            x = self.norm1(x + attention_output)
            x = self.norm2(x + self._feedforward_block(x))
            return x, extra_output
        if extra_output is not None:
            return x, *extra_output
        return x

    @property
    def embed_dim(self):
        return self.mha.embed_dim

# Transformer Encoders -----------------------------------------------------------------------------

class ITransformerEncoder(abc.ABC, L.LightningModule):
    """
    The interface for a transformer encoder block.
    """
    @abc.abstractmethod
    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ) -> Any:
        return NotImplemented

    @property
    @abc.abstractmethod
    def embed_dim(self) -> int:
        return NotImplemented

@export
class TransformerEncoderBlock(ITransformerEncoder, L.LightningModule):
    def __init__(
        self,
        mha: MultiHeadAttention,
        feedforward_dim: int,
        feedforward_activation: Union[L.LightningModule, Callable] = F.gelu,
        norm_first: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.mab = MultiHeadAttentionBlock(
            mha=mha,
            feedforward_dim=feedforward_dim,
            feedforward_activation=feedforward_activation,
            norm_first=norm_first,
            dropout=dropout)
        self.attention_head_mask = nn.Parameter(torch.ones(mha.num_heads), requires_grad=False)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        if attention_head_mask is None:
            attention_head_mask = self.attention_head_mask
        return self.mab.forward(
            x=src,
            y=src,
            attention_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            attention_head_mask=attention_head_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights)

    @property
    def embed_dim(self) -> int:
        return self.mab.embed_dim

    @property
    def num_heads(self) -> int:
        return self.mab.mha.num_heads


@export
class InducedSetAttentionBlock(ITransformerEncoder, L.LightningModule):
    def __init__(
        self,
        mha: MultiHeadAttention,
        num_inducing_points: int,
        feedforward_dim: int,
        feedforward_activation: Union[L.LightningModule, Callable] = F.gelu,
        norm_first: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_inducing_points = num_inducing_points
        self.inducing_points = nn.Parameter(torch.randn(num_inducing_points, mha.embed_dim))
        self.mab1 = MultiHeadAttentionBlock(copy.deepcopy(mha), feedforward_dim, feedforward_activation, norm_first, dropout)
        self.mab2 = MultiHeadAttentionBlock(copy.deepcopy(mha), feedforward_dim, feedforward_activation, norm_first, dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        if src_mask is not None:
            raise Exception("Attention mask not supported for InducedSetAttentionBlock.")
        i = self.inducing_points.unsqueeze(-3).expand(*src.shape[:-2], -1, -1)
        h = self.mab1(i, src, key_padding_mask=src_key_padding_mask)
        return self.mab2(
            src,
            h,
            attention_head_mask=attention_head_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights)

    @property
    def embed_dim(self):
        return self.mab1.mha.embed_dim

    @property
    def num_heads(self):
        return self.mab1.mha.num_heads

@export
class TransformerEncoder(ITransformerEncoder, L.LightningModule):
    def __init__(self, encoder_layer: ITransformerEncoder, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        extra_outputs = []
        output = src
        head_mask = None
        for i, layer in enumerate(cast(Sequence[ITransformerEncoder], self.layers)):
            if attention_head_mask is not None:
                head_mask = attention_head_mask[i]
            output = layer(
                output,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                attention_head_mask=head_mask,
                average_attention_weights=average_attention_weights,
                return_attention_weights=return_attention_weights)
            if isinstance(output, tuple):
                output, *extra_output = output
                if len(extra_output) > 0:
                    extra_outputs.append(extra_output)
        if len(extra_outputs) > 0:
            return output, *zip(*extra_outputs)
        return output

    @property
    def attention_head_mask(self):
        return torch.stack([layer.attention_head_mask for layer in self.layers])

    @attention_head_mask.setter
    def attention_head_mask(self, attention_head_mask):
        for layer, mask in zip(self.layers, attention_head_mask):
            layer.attention_head_mask[:] = mask

    @property
    def embed_dim(self):
        return self.layers[0].embed_dim

    @property
    def num_heads(self):
        return self.layers[0].num_heads

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index):
        return self.layers[index]

# Transformer Decoders -----------------------------------------------------------------------------

class ITransformerDecoder(abc.ABC):
    """
    The interface for a transformer encoder block.
    """
    @abc.abstractmethod
    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        return NotImplemented


@export
class TransformerDecoderBlock(ITransformerDecoder, L.LightningModule):

    def __init__(
        self,
        mha: MultiHeadAttention,
        feedforward_dim: int,
        feedforward_activation: Union[L.LightningModule, Callable] = F.gelu,
        norm_first: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_mha = copy.deepcopy(mha)
        self.cross_mha = copy.deepcopy(mha)
        self.feedforward_dim = feedforward_dim
        self.feedforward_activation = feedforward_activation
        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(self.self_mha.embed_dim)
        self.norm2 = nn.LayerNorm(self.self_mha.embed_dim)
        self.norm3 = nn.LayerNorm(self.self_mha.embed_dim)

        self.feedforward_linear1 = nn.Linear(self.self_mha.embed_dim, feedforward_dim)
        self.feedforward_linear2 = nn.Linear(feedforward_dim, self.self_mha.embed_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def _self_attention_block(
        self,
        target: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        average_attention_weights: bool,
        return_attention_weights: bool
    ) -> Tuple[torch.Tensor, Optional[List[Any]]]:
        attention_output = self.self_mha(
            target,
            target,
            target,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights)
        if isinstance(attention_output, tuple):
            attention_output, *extra = attention_output
        else:
            extra = None
        return self.dropout1(attention_output), extra

    def _cross_attention_block(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        average_attention_weights: bool,
        return_attention_weights: bool
    ) -> Tuple[torch.Tensor, Optional[List[Any]]]:
        attention_output = self.cross_mha(
            target,
            memory,
            memory,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights)
        if isinstance(attention_output, tuple):
            attention_output, *extra = attention_output
        else:
            extra = None
        return self.dropout2(attention_output), extra

    def _feedforward_block(self, x: torch.Tensor):
        return self.dropout3(
            self.feedforward_linear2(self.feedforward_activation(self.feedforward_linear1(x)))
        )

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        x = target
        if self.norm_first:
            attention_output, extra_output_sa = self._self_attention_block(
                self.norm1(x),
                target_mask,
                target_key_padding_mask,
                average_attention_weights,
                return_attention_weights)
            x = x + attention_output
            attention_output, extra_output_ca = self._cross_attention_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                average_attention_weights,
                return_attention_weights)
            x = x + attention_output
            x = x + self._feedforward_block(self.norm3(x))
        else:
            attention_output, extra_output_sa = self._self_attention_block(
                x,
                target_mask,
                target_key_padding_mask,
                average_attention_weights,
                return_attention_weights)
            x = self.norm1(x + attention_output)
            attention_output, extra_output_ca = self._cross_attention_block(
                x,
                memory,
                memory_mask,
                memory_key_padding_mask,
                average_attention_weights,
                return_attention_weights)
            x = self.norm2(x + attention_output)
            x = self.norm3(x + self._feedforward_block(x))
        extra_output = ()
        if extra_output_sa is not None:
            extra_output += tuple(extra_output_sa)
        if extra_output_ca is not None:
            extra_output += tuple(extra_output_ca)
        if len(extra_output) > 0:
            return x, *extra_output
        return x


@export
class TransformerDecoder(ITransformerDecoder, L.LightningModule):
    def __init__(self, decoder_layer: TransformerDecoderBlock, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        extra_outputs = []
        output = target
        for layer in cast(Sequence[TransformerDecoderBlock], self.layers):
            output = layer(
                target=output,
                memory=memory,
                target_mask=target_mask,
                memory_mask=memory_mask,
                target_key_padding_mask=target_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                average_attention_weights=average_attention_weights,
                return_attention_weights=return_attention_weights)
            if isinstance(output, tuple):
                output, *extra_output = output
                if len(extra_output) > 0:
                    extra_outputs.append(extra_output)
        if len(extra_outputs) > 0:
            return output, *zip(*extra_outputs)
        return output

    @property
    def embed_dim(self):
        return self.layers[0].embed_dim

    def __len__(self):
        return len(self.layers)


@export
class ConditionedInducedSetAttentionBlock(TransformerDecoderBlock):
    def __init__(
        self,
        mha: MultiHeadAttention,
        num_inducing_points: int,
        feedforward_dim: int,
        feedforward_activation: Union[L.LightningModule, Callable] = F.gelu,
        norm_first: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__(mha, feedforward_dim, feedforward_activation, norm_first, dropout)
        self.num_inducing_points = num_inducing_points
        self.inducing_point_predictor = nn.Linear(mha.embed_dim, mha.embed_dim*num_inducing_points)
        self.mab1 = MultiHeadAttentionBlock(copy.deepcopy(mha), feedforward_dim, feedforward_activation, norm_first, dropout)
        self.mab2 = MultiHeadAttentionBlock(copy.deepcopy(mha), feedforward_dim, feedforward_activation, norm_first, dropout)

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        target_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False
    ):
        if target_mask is not None:
            raise Exception(f"target_mask not supported for {self.__class__}")
        if memory_mask is not None:
            raise Exception(f"memory_mask not supported for {self.__class__}")
        if memory_key_padding_mask is not None:
            raise Exception(f"Memory key padding mask not supported for {self.__class__}")
        i = self.inducing_point_predictor(memory).view(*memory.shape[:-1], self.num_inducing_points, -1)
        h = self.mab1(i, target, attention_mask=target_key_padding_mask)
        return self.mab2(target, h, average_attention_weights=average_attention_weights, return_attention_weights=return_attention_weights)

    @property
    def embed_dim(self):
        return self.mab1.embed_dim

# Miscellaneous

@export
class SampleSet(L.LightningModule):
    def __init__(self, embed_dim: int, max_set_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_set_size = max_set_size
        self.mu = nn.Parameter(torch.randn(max_set_size, embed_dim))
        self.sigma = nn.Parameter(torch.abs(torch.randn(max_set_size, embed_dim)))

    def forward(self, n: torch.Tensor, masked: bool = False):
        batch_size = n.size(0)
        torch.randperm()
