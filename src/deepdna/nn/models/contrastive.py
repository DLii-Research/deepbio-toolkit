import copy
from itertools import combinations
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Sequence, Union

from . import LightningModuleWithHyperparameters
from .. import layers
from ...data.vocabularies import dna, Vocabulary
from ..._utils import export

# Contrastive Pretraining --------------------------------------------------------------------------

@export
class ContrastivePretrainingModel(L.LightningModule):
    def __init__(
        self,
        encoders: Union[Sequence[nn.Module], Dict[Any, nn.Module]],
        projection_dims: Union[int, Sequence[int], Dict[Any, int]],
        embed_dim: int,
        shared_projections: bool = False,
        max_temp: float = 100.0
    ):
        super().__init__()
        if shared_projections:
            w = [nn.Linear(projection_dims, embed_dim, bias=False)]*len(encoders) # type: ignore
        else:
            w = [nn.Linear(d, embed_dim, bias=False) for d in projection_dims] # type: ignore
        if isinstance(encoders, dict):
            self.encoders = nn.ModuleDict(encoders)
            self.w = nn.ModuleDict(dict(zip(self.encoders.keys(), w)))
        else:
            self.encoders = nn.ModuleList(tuple(encoders))
            self.w = nn.ModuleList(w)
        self.embed_dim = embed_dim
        self.max_temp = max_temp
        self.shared_projections = shared_projections
        self.t = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        if isinstance(batch, dict):
            features = [
                self.w[key](self.encoders[key](x)) for key, x in batch.items()
            ]
        else:
            features = [
                w(encoder(x)) for encoder, w, x in zip(self.encoders, self.w, batch) # type: ignore
            ]
        embeddings = [F.normalize(f, p=2, dim=1) for f in features]
        if isinstance(batch, dict):
            return dict(zip(batch.keys(), embeddings))
        return embeddings

    def _step(self, stage: str, batch):
        embeddings = self(batch)
        comparisons = list(combinations(embeddings, 2))
        loss = 0.0
        for a, b in comparisons:
            logits = torch.tensordot(a, b.T, 1)
            labels = torch.arange(a.size(0)).to(logits.device)
            loss_a = F.cross_entropy(logits * torch.exp(self.t), labels)
            loss_b = F.cross_entropy(logits.T * torch.exp(self.t), labels)
            loss += (loss_a + loss_b) / 2
        loss /= len(comparisons)
        self.log(f"{stage}/loss", loss)
        return loss

    def training_step(self, batch):
        return self._step("train", batch)

    def validation_step(self, batch):
        return self._step("val", batch)

    def test_step(self, batch):
        return self._step("test", batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# Encoders -----------------------------------------------------------------------------------------

@export
class DnaTransformerModel(LightningModuleWithHyperparameters):
    def __init__(
        self,
        transformer_encoder: layers.TransformerEncoder,
        kmer: int = 1,
        kmer_stride: int = 1
    ):
        super().__init__()
        self.transformer_encoder = transformer_encoder
        self.kmer = kmer
        self.kmer_stride = kmer_stride
        self.class_token = nn.Parameter(torch.randn(self.transformer_encoder.embed_dim))
        self.vocabulary = Vocabulary(dna(kmer))

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False,
        **kwargs
    ):
        class_tokens = self.class_token.expand(*src.shape[:-2], 1, -1)
        src = torch.cat((class_tokens, src), -2)
        output = self.transformer_encoder(
            src,
            src_key_padding_mask=src_key_padding_mask,
            attention_head_mask=attention_head_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights,
            **kwargs
        )
        output, *extra = output if isinstance(output, tuple) else (output,)
        class_tokens = output.select(-2, 0)
        output_tokens = output.narrow(-2, 1, src.shape[-2] - 1)
        return class_tokens, output_tokens, *extra


@export
class AmpliconSampleTransformerModel(LightningModuleWithHyperparameters):
    def __init__(
        self,
        transformer_encoder: layers.TransformerEncoder,
        dna_encoder: nn.Module,
        vocabulary: Vocabulary,
    ):
        super().__init__()
        self.transformer_encoder = transformer_encoder
        self.dna_encoder = dna_encoder
        self.vocabulary = vocabulary

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        attention_head_mask: Optional[torch.Tensor] = None,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False,
        **kwargs
    ):
        if not src.is_floating_point() and self.dna_encoder is not None:
            # Needs to be encoded.
            pad_id = self.vocabulary["[PAD]"]
            if src_key_padding_mask is None:
                src_key_padding_mask = torch.all(src == pad_id, -1)
            with torch.no_grad():
                src = self.dna_encoder(src)
        output = self.transformer_encoder(
            src,
            src_key_padding_mask=src_key_padding_mask,
            attention_head_mask=attention_head_mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights,
            **kwargs
        )
        output, *extra = output if isinstance(output, tuple) else (output,)
        class_tokens = output.select(-2, 0)
        output_tokens = output.narrow(-2, 1, src.shape[-2] - 1)
        return class_tokens, output_tokens, *extra