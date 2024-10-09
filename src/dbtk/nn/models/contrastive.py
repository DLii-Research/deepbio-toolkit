import copy
from itertools import combinations
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Sequence, Union

from .. import layers
from ...data.vocabularies import dna, Vocabulary
from ..._utils import export

# Contrastive Pretraining --------------------------------------------------------------------------

@export
class ContrastivePretrainingModel(L.LightningModule):
    def __init__(
        self,
        encoder_a: nn.Module,
        encoder_b: nn.Module,
        embed_dim_a: int,
        embed_dim_b: int,
        projection_dim: Optional[int] = None,
        shared_projections: Optional[bool] = None,
        max_temp: float = 100.0
    ):
        super().__init__()

        if encoder_a is encoder_b:
            assert shared_projections is not False, "Shared projections are required for one encoder."
            shared_projections = True
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.embed_dim_a = embed_dim_a
        self.embed_dim_b = embed_dim_b
        self.shared_projections = shared_projections
        self.projection_dim = projection_dim if projection_dim is not None else min(embed_dim_a, embed_dim_b)
        self.max_temperature = max_temp
        # Parameters
        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.w_a = self.w_b = nn.Linear(self.embed_dim_a, self.projection_dim, bias=False)
        if not self.shared_projections:
            self.w_b = nn.Linear(self.embed_dim_b, self.projection_dim, bias=False)

        for stage in ["training", "validation", "testing"]:
            setattr(self, f"{stage}_step", partial(self._step, stage=stage))

    def forward(self, batch):
        a, b = batch
        return torch.stack([
            F.normalize(self.w_a(self.encode_a(a)), p=2, dim=-1),
            F.normalize(self.w_b(self.encode_b(b)), p=2, dim=-1)
        ])

    def _step(self, batch, stage: str,):
        a, b = self.all_gather(self(batch), sync_grads=True)
        logits = torch.tensordot(a, b.transpose(-1, -2), a.ndim - 1)
        labels = torch.arange(a.size(0), device=logits.device)
        loss_a = F.cross_entropy(logits * torch.exp(self.temperature), labels)
        loss_b = F.cross_entropy(logits.transpose(-1, -2) * torch.exp(self.temperature), labels)
        loss += (loss_a + loss_b) / 2
        accuracy = torch.sum(
            (torch.argmax(logits, dim=-1) == labels) + (torch.argmax(logits, dim=-2) == labels)
        ) / 2.0 / x.size(0)
        self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self.temperature.data = torch.clamp(self.temperature, 0.0, self.max_temp)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

# @export
# class ContrastivePretrainingModel(L.LightningModule):
#     def __init__(
#         self,
#         encoders: Union[Sequence[nn.Module], Dict[Any, nn.Module]],
#         projection_dims: Union[int, Sequence[int], Dict[Any, int]],
#         embed_dim: Optional[int] = None,
#         shared_projections: bool = False,
#         max_temp: float = 100.0
#     ):
#         super().__init__()
#         if len(set(map(id, encoders))) == 1: # All the same encoder, use shared projections
#             shared_projections = True
#         if shared_projections:
#             w = [nn.Linear(projection_dims, embed_dim, bias=False)]*len(encoders) # type: ignore
#         else:
#             w = [nn.Linear(d, embed_dim, bias=False) for d in projection_dims] # type: ignore
#         if isinstance(encoders, dict):
#             self.encoders = nn.ModuleDict(encoders)
#             self.w = nn.ModuleDict(dict(zip(self.encoders.keys(), w)))
#         else:
#             self.encoders = nn.ModuleList(tuple(encoders))
#             self.w = nn.ModuleList(w)
#         self.embed_dim = embed_dim
#         self.max_temp = max_temp
#         self.shared_projections = shared_projections
#         self.t = nn.Parameter(torch.tensor(1.0))

#     def forward(self, batch):
#         if isinstance(batch, dict):
#             features = [
#                 self.w[key](self.encoders[key](x)) for key, x in batch.items()
#             ]
#         else:
#             features = [
#                 w(encoder(x)) for encoder, w, x in zip(self.encoders, self.w, batch) # type: ignore
#             ]
#         embeddings = [F.normalize(f, p=2, dim=1) for f in features]
#         if isinstance(batch, dict):
#             return dict(zip(batch.keys(), embeddings))
#         return embeddings

#     def _step(self, stage: str, batch):
#         embeddings = self(batch)
#         comparisons = list(combinations(embeddings, 2))
#         loss = 0.0
#         accuracy = 0.0
#         for a, b in comparisons:
#             logits = torch.tensordot(a, b.T, 1)
#             labels = torch.arange(a.size(0)).to(logits.device)
#             loss_a = F.cross_entropy(logits * torch.exp(self.t), labels)
#             loss_b = F.cross_entropy(logits.transpose(-1, -2) * torch.exp(self.t), labels)
#             loss += (loss_a + loss_b) / 2
#             torch.sum(torch.argmax(logits, dim=-2) == labels)
#             accuracy += torch.sum(
#                 (torch.argmax(logits, dim=-1) == labels) + (torch.argmax(logits, dim=-2) == labels)
#             ) / 2.0 / x.size(0)
#         loss /= len(comparisons)
#         accuracy /= len(comparisons)
#         self.log(f"{stage}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
#         self.log(f"{stage}/accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
#         return loss

#     def training_step(self, batch):
#         return self._step("train", batch)

#     def validation_step(self, batch):
#         return self._step("validation", batch)

#     def test_step(self, batch):
#         return self._step("test", batch)

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=1e-4)